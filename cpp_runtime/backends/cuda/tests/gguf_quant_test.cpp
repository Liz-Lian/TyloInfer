#include "../kernels/gguf.h"
#include "../kernels/gguf_compact.h"
#include "tyloinfer/gguf_quant.h"
#include "ggml-quants.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<typename T> struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(std::size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T))); }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
};

void check_type(ggml_type type, int columns, int rows=5) {
    const auto stride = mfq::gguf::row_bytes(type, columns);
    std::vector<std::uint8_t> bytes(rows * stride);
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = (i * 29 + i / 7 + 17) & 63;
    if(type==GGML_TYPE_IQ3_XXS || type==GGML_TYPE_IQ3_S || type==GGML_TYPE_IQ2_XXS ||
            type==GGML_TYPE_IQ2_XS || type==GGML_TYPE_IQ2_S) {
        // Exercise every sign/code bit without creating NaNs in the block's
        // FP16 scale. Restricting every byte to six bits misses sign patterns.
        for(std::size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(i*29+i/7+17);
        const uint16_t scale=0x3000;
        for(std::size_t i=0;i<bytes.size();i+=ggml_type_size(type))std::memcpy(bytes.data()+i,&scale,sizeof(scale));
    }
    std::vector<float> reference(rows * columns), decoded(reference.size());
    const auto* traits = ggml_get_type_traits(type);
    for (int r = 0; r < rows; ++r) {
        if (type == GGML_TYPE_Q8_K)
            dequantize_row_q8_K(reinterpret_cast<const block_q8_K*>(bytes.data() + r * stride),
                reference.data() + r * columns, columns);
        else {
            if (!traits->to_float) throw std::runtime_error("missing ggml reference decoder");
            traits->to_float(bytes.data() + r * stride, reference.data() + r * columns, columns);
        }
    }
    DeviceArray<std::uint8_t> packed(bytes.size());
    DeviceArray<float> output(reference.size());
    check(cudaMemcpy(packed.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    mfq::cuda::gguf::dequantize_rows(type, packed.data, stride, rows, columns, output.data, false, nullptr);
    check(cudaMemcpy(decoded.data(), output.data, decoded.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(decoded[i]) || std::abs(decoded[i] - reference[i]) >
                1e-6f * std::max(1.f, std::abs(reference[i])))
            throw std::runtime_error(std::string(ggml_type_name(type)) + " decoder mismatch at " + std::to_string(i));
    }
    std::vector<std::int64_t> ids{3, 0, 3, rows - 1, -1, rows};
    DeviceArray<std::int64_t> device_ids(ids.size());
    DeviceArray<float> embedding(ids.size() * columns);
    check(cudaMemcpy(device_ids.data, ids.data(), ids.size() * sizeof(ids[0]), cudaMemcpyHostToDevice));
    mfq::cuda::gguf::embedding_rows(type, packed.data, stride, rows, columns,
        device_ids.data, ids.size(), embedding.data, nullptr);
    std::vector<float> embedded(ids.size() * columns);
    check(cudaMemcpy(embedded.data(), embedding.data, embedded.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < embedded.size(); ++i) {
        const auto row = ids[i / columns];
        const auto expected = row >= 0 && row < rows ? reference[row * columns + i % columns] : 0.f;
        if (std::abs(embedded[i] - expected) > 1e-6f * std::max(1.f, std::abs(expected)))
            throw std::runtime_error("GGUF embedding mismatch");
    }
    // A captured PLE decoder must read updated row packets on every replay.
    using mfq::cuda::gguf::StagedRow;
    const auto staged_stride = (stride + 7) & ~std::uint64_t(7);
    std::vector<std::uint8_t> packet(2 * sizeof(StagedRow) + 2 * staged_stride);
    auto* descriptors = reinterpret_cast<StagedRow*>(packet.data());
    for (int i = 0; i < 2; ++i)
        descriptors[i] = {2 * sizeof(StagedRow) + i * staged_stride, std::int32_t(type), 0};
    DeviceArray<std::uint8_t> device_packet(packet.size());
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    mfq::cuda::gguf::staged_rows(device_packet.data, 2, columns, output.data, false, stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    for (const int first_row : {3, 0}) {
        std::memcpy(packet.data() + descriptors[0].offset, bytes.data() + first_row * stride, stride);
        std::memcpy(packet.data() + descriptors[1].offset, bytes.data() + stride, stride);
        check(cudaMemcpyAsync(device_packet.data, packet.data(), packet.size(), cudaMemcpyHostToDevice, stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(decoded.data(), output.data, 2 * columns * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 2 * columns; ++i) {
            const auto expected = reference[(i < columns ? first_row : 1) * columns + i % columns];
            if (std::abs(decoded[i] - expected) > 1e-6f * std::max(1.f, std::abs(expected)))
                throw std::runtime_error("GGUF row packet graph replay mismatch");
        }
    }
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
    for (const int tokens : {1, 2, 3, 4, 5, 6, 7, 8}) {
        std::vector<float> x(tokens * columns), y(tokens * rows);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.13f) * 0.1f;
        DeviceArray<float> dx(x.size()), dy(y.size());
        check(cudaMemcpy(dx.data, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        mfq::cuda::gguf::matvec(type, packed.data, stride, rows, columns,
            dx.data, tokens, dy.data, nullptr);
        check(cudaMemcpy(y.data(), dy.data, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int t = 0; t < tokens; ++t) for (int r = 0; r < rows; ++r) {
            double expected = 0., magnitude = 0.;
            for (int k = 0; k < columns; ++k) {
                const double product = double(reference[r * columns + k]) * x[t * columns + k];
                expected += product; magnitude += std::abs(product);
            }
            if (!std::isfinite(y[t * rows + r]) ||
                    std::abs(y[t * rows + r] - expected) > 2e-6 * magnitude + 1e-6)
                throw std::runtime_error(std::string(ggml_type_name(type)) + " projection mismatch");
        }
        DeviceArray<std::uint8_t> quantized(tokens * mfq::cuda::gguf::q8_row_bytes(columns));
        mfq::cuda::gguf::quantize_input(dx.data, tokens, columns, quantized.data, nullptr);
        mfq::cuda::gguf::matvec_q8(type, packed.data, stride, rows, columns,
            quantized.data, tokens, dy.data, nullptr);
        check(cudaMemcpy(y.data(), dy.data, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if(mfq::cuda::gguf::supports_interleaved_activation(type,tokens)) {
            DeviceArray<std::uint8_t> position_major(mfq::cuda::gguf::interleaved_activation_bytes(tokens,columns));
            mfq::cuda::gguf::quantize_input_interleaved(dx.data,false,tokens,columns,position_major.data,nullptr,nullptr);
            mfq::cuda::gguf::matvec_interleaved(type,packed.data,stride,rows,columns,position_major.data,tokens,dy.data,false,nullptr);
            std::vector<float> actual(y.size());
            check(cudaMemcpy(actual.data(),dy.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
            if(std::memcmp(actual.data(),y.data(),y.size()*sizeof(float)))
                throw std::runtime_error("position-major projection changed complete quantized code/sign coverage");
        }

        {
            // The fused store must perform exactly the old final conversion,
            // including the cooperative row layout, odd N and token tails.
            DeviceArray<ggml_fp16_t> half_output(y.size());
            mfq::cuda::gguf::matvec_q8_f16(type,packed.data,stride,rows,columns,
                quantized.data,tokens,half_output.data,nullptr);
            std::vector<ggml_fp16_t> got(y.size());
            check(cudaMemcpy(got.data(),half_output.data,got.size()*sizeof(got[0]),cudaMemcpyDeviceToHost));
            for(std::size_t i=0;i<got.size();++i)if(got[i]!=ggml_fp32_to_fp16(y[i]))
                throw std::runtime_error(std::string(ggml_type_name(type))+" fused FP16 projection changed output rounding");
            std::vector<ggml_fp16_t> half_input(x.size());std::vector<float> rounded(x.size());
            for(std::size_t i=0;i<x.size();++i){half_input[i]=ggml_fp32_to_fp16(x[i]);rounded[i]=ggml_fp16_to_fp32(half_input[i]);}
            DeviceArray<ggml_fp16_t> input_half(x.size());DeviceArray<float> input_rounded(x.size());
            DeviceArray<std::uint8_t> half_q(tokens*mfq::cuda::gguf::q8_row_bytes(columns));
            DeviceArray<std::uint8_t> float_q(tokens*mfq::cuda::gguf::q8_row_bytes(columns));
            check(cudaMemcpy(input_half.data,half_input.data(),half_input.size()*sizeof(half_input[0]),cudaMemcpyHostToDevice));
            check(cudaMemcpy(input_rounded.data,rounded.data(),rounded.size()*sizeof(float),cudaMemcpyHostToDevice));
            mfq::cuda::gguf::quantize_input_f16(input_half.data,tokens,columns,half_q.data,nullptr);
            mfq::cuda::gguf::quantize_input(input_rounded.data,tokens,columns,float_q.data,nullptr);
            std::vector<std::uint8_t> a(tokens*mfq::cuda::gguf::q8_row_bytes(columns)),b(a.size());
            check(cudaMemcpy(a.data(),half_q.data,a.size(),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(b.data(),float_q.data,b.size(),cudaMemcpyDeviceToHost));
            if(a!=b)throw std::runtime_error("fused FP16 activation quantization changed Q8 bytes");
        }
        if(type==GGML_TYPE_Q4_K || type==GGML_TYPE_Q5_K || type==GGML_TYPE_Q6_K) {
            std::vector<block_q8_1> blocks(tokens*columns/32);
            check(cudaMemcpy(blocks.data(),quantized.data,blocks.size()*sizeof(block_q8_1),cudaMemcpyDeviceToHost));
            for(int t=0;t<tokens;++t)for(int r=0;r<rows;++r) {
                double expected=0,magnitude=0;
                for(int k=0;k<columns;++k) {
                    const auto& block=blocks[t*(columns/32)+k/32];
                    const double product=double(reference[r*columns+k])*ggml_fp16_to_fp32(block.d)*block.qs[k%32];
                    expected+=product;magnitude+=std::abs(product);
                }
                if(std::abs(y[t*rows+r]-expected)>2e-6*magnitude+1e-6)
                    throw std::runtime_error(std::string(ggml_type_name(type))+" prepared integer dot changed decoded weights or scales");
            }
        }
        for (int t = 0; t < tokens; ++t) for (int r = 0; r < rows; ++r) {
            double expected = 0., magnitude = 0.;
            for (int k = 0; k < columns; ++k) {
                const double product = double(reference[r * columns + k]) * x[t * columns + k];
                expected += product; magnitude += std::abs(product);
            }
            if (!std::isfinite(y[t * rows + r]) || std::abs(y[t * rows + r] - expected) > .008 * magnitude + 1e-6)
                throw std::runtime_error(std::string(ggml_type_name(type)) + " integer projection mismatch");
        }
    }
    // Distinct banks, repeated experts, missing addresses and invalid routes.
    // Check both shared Gate/Up inputs and the per-route Down input layout.
    std::vector<std::uint8_t> banks(3 * bytes.size());
    for (int expert = 0; expert < 3; ++expert) for (int row = 0; row < rows; ++row)
        std::memcpy(banks.data() + expert * bytes.size() + row * stride,
            bytes.data() + (row + expert) % rows * stride, stride);
    DeviceArray<std::uint8_t> device_banks(banks.size());
    check(cudaMemcpy(device_banks.data, banks.data(), banks.size(), cudaMemcpyHostToDevice));
    std::vector<std::uint64_t> addresses{reinterpret_cast<std::uint64_t>(device_banks.data),
        reinterpret_cast<std::uint64_t>(device_banks.data + bytes.size()),
        reinterpret_cast<std::uint64_t>(device_banks.data + 2 * bytes.size()), 0};
    DeviceArray<std::uint64_t> device_addresses(addresses.size());
    check(cudaMemcpy(device_addresses.data, addresses.data(), addresses.size() * sizeof(addresses[0]), cudaMemcpyHostToDevice));
    for (const int tokens : {1, 2, 3, 4, 5, 6, 7, 8}) for (const bool routed : {false, true}) {
        constexpr int routes = 3;
        std::vector<std::int64_t> selected(tokens * routes);
        const std::int64_t pattern[]{0, 2, 0, 1, 3, -1, 4, 2, 1, 0, 1, 2};
        for(std::size_t i=0;i<selected.size();++i)selected[i]=pattern[i%12];
        std::vector<float> x(tokens * (routed ? routes : 1) * columns), y(tokens * routes * rows);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::cos(float(i) * .17f) * .1f;
        DeviceArray<std::int64_t> selected_device(selected.size());
        DeviceArray<float> dx(x.size()), dy(y.size());
        check(cudaMemcpy(selected_device.data, selected.data(), selected.size() * sizeof(selected[0]), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dx.data, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        DeviceArray<std::uint8_t> quantized(x.size() / columns * mfq::cuda::gguf::q8_row_bytes(columns));
        mfq::cuda::gguf::quantize_input(dx.data, x.size() / columns, columns, quantized.data, nullptr);
        mfq::cuda::gguf::routed_matvec_q8(type, device_addresses.data, stride, addresses.size(), rows,
            columns, selected_device.data, tokens, routes, quantized.data, routed, dy.data, nullptr);
        check(cudaMemcpy(y.data(), dy.data, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
        DeviceArray<mfq::cuda::gguf::RouteGroup> groups(selected.size()+1);
        mfq::cuda::gguf::prepare_route_groups(selected_device.data,selected.size(),tokens,groups.data,nullptr);
        if(tokens>4) {
            std::vector<mfq::cuda::gguf::RouteGroup> plan(selected.size()+1);
            check(cudaMemcpy(plan.data(),groups.data,plan.size()*sizeof(plan[0]),cudaMemcpyDeviceToHost));
            std::vector<int> seen(selected.size());
            for(int g=1;g<=plan[0].count;++g) {
                if(plan[g].count<1 || plan[g].count>4)throw std::runtime_error("routed reuse tile exceeded four entries");
                for(int t=0;t<plan[g].count;++t) {
                    const int slot=plan[g].slots[t];
                    if(slot<0 || slot>=int(selected.size()) || selected[slot]!=plan[g].expert || seen[slot]++)
                        throw std::runtime_error("routed grouping lost or duplicated a slot");
                }
            }
            if(std::any_of(seen.begin(),seen.end(),[](int count){return count!=1;}))
                throw std::runtime_error("routed grouping omitted an input row");
        }
        check(cudaMemset(dy.data,0xff,y.size()*sizeof(float)));
        mfq::cuda::gguf::grouped_routed_matvec_q8(type,device_addresses.data,stride,addresses.size(),rows,columns,
            selected_device.data,groups.data,tokens,routes,quantized.data,routed,dy.data,nullptr);
        std::vector<float> grouped_output(y.size());
        check(cudaMemcpy(grouped_output.data(),dy.data,grouped_output.size()*sizeof(float),cudaMemcpyDeviceToHost));
        if(type>=GGML_TYPE_TQ2_T && type<=GGML_TYPE_TQK4) {
            // The raw single-entry kernel factors block scales. Check it
            // against independently decoded weights and the actual Q8 input,
            // keeping activation quantization error out of this bound.
            std::vector<block_q8_1> packets(x.size()/32);
            check(cudaMemcpy(packets.data(),quantized.data,packets.size()*sizeof(packets[0]),cudaMemcpyDeviceToHost));
            for(int slot=0;slot<tokens*routes;++slot)for(int row=0;row<rows;++row) {
                const auto expert=selected[slot];double expected=0,magnitude=0;
                if(expert>=0 && expert<3)for(int k=0;k<columns;++k) {
                    const auto& packet=packets[(routed?slot:slot/routes)*(columns/32)+k/32];
                    const double product=double(reference[((row+expert)%rows)*columns+k])*
                        ggml_fp16_to_fp32(packet.d)*packet.qs[k%32];
                    expected+=product;magnitude+=std::abs(product);
                }
                if(!std::isfinite(y[slot*rows+row]) || std::abs(double(y[slot*rows+row])-expected)>2e-6*magnitude+1e-6)
                    throw std::runtime_error("trellis factored block scales exceeded independent Q8 dot bound");
            }
            // Native Q8 staging and decoded FP32 slices must agree, including
            // mixed group sizes, absent experts and odd K/N tails.
            if(!mfq::cuda::gguf::try_compact_routed_project(mfq::cuda::gguf::CompactFormat(type),
                    device_addresses.data,stride,addresses.size(),rows,columns,selected_device.data,
                    groups.data,tokens,routes,quantized.data,routed,dy.data,nullptr,false,nullptr,0,nullptr,true,false))
                throw std::runtime_error("compact Q8-staging reference unavailable");
            std::vector<float> native(y.size());
            check(cudaMemcpy(native.data(),dy.data,native.size()*sizeof(float),cudaMemcpyDeviceToHost));
            if(std::memcmp(native.data(),grouped_output.data(),native.size()*sizeof(float)))
                throw std::runtime_error("decoded activation slices changed exact FP32 projection bits");
            // Changing lanes per row changes the FP32 reduction tree. Compare
            // with the original layout against a sum-of-products error bound,
            // including odd row tails and each circular trellis encoding.
            if(!mfq::cuda::gguf::try_compact_routed_project(mfq::cuda::gguf::CompactFormat(type),
                    device_addresses.data,stride,addresses.size(),rows,columns,selected_device.data,
                    groups.data,tokens,routes,quantized.data,routed,dy.data,nullptr,false,nullptr,0,nullptr,true,true,false))
                throw std::runtime_error("original trellis row layout unavailable");
            std::vector<float> original(y.size());
            check(cudaMemcpy(original.data(),dy.data,original.size()*sizeof(float),cudaMemcpyDeviceToHost));
            for(int slot=0;slot<tokens*routes;++slot)for(int row=0;row<rows;++row) {
                const auto expert=selected[slot];double magnitude=0;
                if(expert>=0 && expert<3)for(int k=0;k<columns;++k)
                    magnitude+=std::abs(double(reference[((row+expert)%rows)*columns+k])*
                        x[(routed?slot:slot/routes)*columns+k]);
                const auto at=slot*rows+row;
                if(!std::isfinite(grouped_output[at]) || !std::isfinite(original[at]) ||
                        std::abs(double(grouped_output[at])-original[at])>2e-6*magnitude+1e-6)
                    throw std::runtime_error("narrow trellis row layout exceeded FP32 accumulation bound");
            }
        }
        if(!routed && tokens<=8) {
            auto up_addresses=addresses;std::rotate(up_addresses.begin(),up_addresses.begin()+1,up_addresses.end());
            DeviceArray<uint64_t> device_up(up_addresses.size());
            check(cudaMemcpy(device_up.data,up_addresses.data(),up_addresses.size()*sizeof(uint64_t),cudaMemcpyHostToDevice));
            DeviceArray<float> gu(y.size()),uu(y.size()),up_reference(y.size());
            check(cudaMemset(gu.data,0xff,y.size()*sizeof(float)));
            check(cudaMemset(uu.data,0xff,y.size()*sizeof(float)));
            mfq::cuda::gguf::routed_matvec_q8(type,device_up.data,stride,addresses.size(),rows,columns,
                selected_device.data,tokens,routes,quantized.data,false,up_reference.data,nullptr);
            const bool used=mfq::cuda::gguf::try_gate_up_matvec_q8(type,device_addresses.data,device_up.data,stride,stride,
                addresses.size(),rows,columns,selected_device.data,groups.data,tokens,routes,quantized.data,
                gu.data,uu.data,nullptr);
            if(used) {
                std::vector<float> actual(y.size()),expected(y.size());
                check(cudaMemcpy(actual.data(),gu.data,y.size()*sizeof(float),cudaMemcpyDeviceToHost));
                const auto equal_projection=[&](const std::vector<float>& actual,const std::vector<float>& expected,
                        const std::vector<uint64_t>& table) {
                    if(type<GGML_TYPE_TQ2_T || type>GGML_TYPE_TQK4)
                        return std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0;
                    // Trellis uses a different reduction layout for grouped
                    // activations, preserving the same original FP16 values.
                    // Apply the flat/grouped accumulation bound below.
                    for(int slot=0;slot<tokens*routes;++slot)for(int row=0;row<rows;++row) {
                        const auto expert=selected[slot];
                        const auto pointer=expert>=0 && expert<int64_t(table.size())?table[expert]:0;
                        const auto found=std::find(addresses.begin(),addresses.end(),pointer);
                        const auto bank=found-addresses.begin();double magnitude=0;
                        if(pointer && bank<3)for(int k=0;k<columns;++k)
                            magnitude+=std::abs(double(reference[((row+bank)%rows)*columns+k])*x[(slot/routes)*columns+k]);
                        const auto at=slot*rows+row;
                        if(!std::isfinite(actual[at]) || !std::isfinite(expected[at]) ||
                                std::abs(double(actual[at])-expected[at])>2e-6*magnitude+1e-6) {
                            std::fprintf(stderr,"trellis projection mismatch type=%d M=%d row=%d slot=%d actual=%.9g expected=%.9g magnitude=%.9g\n",
                                int(type),tokens,row,slot,actual[at],expected[at],magnitude);
                            return false;
                        }
                    }
                    return true;
                };
                if(!equal_projection(actual,y,addresses))
                    throw std::runtime_error(std::string(ggml_type_name(type))+" fused Gate exceeded reference accumulation bound");
                const auto saved_gate=actual;
                check(cudaMemcpy(actual.data(),uu.data,y.size()*sizeof(float),cudaMemcpyDeviceToHost));
                check(cudaMemcpy(expected.data(),up_reference.data,y.size()*sizeof(float),cudaMemcpyDeviceToHost));
                if(!equal_projection(actual,expected,up_addresses))
                    throw std::runtime_error(std::string(ggml_type_name(type))+" fused Up changed independent pointer table or accumulation");
                if(rows%32==0)for(bool half:{false,true})for(float limit:{0.f,.01f}) {
                    DeviceArray<uint8_t> wanted(selected.size()*mfq::cuda::gguf::q8_row_bytes(rows)),packed_hidden(selected.size()*mfq::cuda::gguf::q8_row_bytes(rows));
                    mfq::cuda::gguf::quantize_moe_hidden(gu.data,uu.data,selected.size(),rows,half,limit,wanted.data,nullptr);
                    if(!mfq::cuda::gguf::try_gate_up_hidden_q8(type,device_addresses.data,device_up.data,stride,stride,
                            addresses.size(),rows,columns,selected_device.data,groups.data,tokens,routes,quantized.data,
                            packed_hidden.data,half,limit,nullptr))continue; // Independently optional epilogue. Generator is tested separately.
                    std::vector<uint8_t> a(selected.size()*mfq::cuda::gguf::q8_row_bytes(rows)),b(a.size());
                    check(cudaMemcpy(a.data(),packed_hidden.data,a.size(),cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(b.data(),wanted.data,b.size(),cudaMemcpyDeviceToHost));
                    if(a!=b)throw std::runtime_error(std::string(ggml_type_name(type))+" fused hidden changed Q8 bytes or FP16/clamp rounding");
                }
                // A missing Gate or Up may independently retain an earlier pass.
                auto absent=up_addresses;std::fill(absent.begin(),absent.end(),0);
                check(cudaMemcpy(device_up.data,absent.data(),absent.size()*sizeof(uint64_t),cudaMemcpyHostToDevice));
                mfq::cuda::gguf::try_gate_up_matvec_q8(type,device_up.data,device_addresses.data,stride,stride,
                    addresses.size(),rows,columns,selected_device.data,groups.data,tokens,routes,quantized.data,
                    gu.data,uu.data,nullptr,true);
                check(cudaMemcpy(actual.data(),gu.data,y.size()*sizeof(float),cudaMemcpyDeviceToHost));
                if(std::memcmp(actual.data(),saved_gate.data(),y.size()*sizeof(float)))
                    throw std::runtime_error("fused Gate did not preserve its missing expert rows");
            }
        }
        // Resident and streamed experts share one result buffer. The second
        // pass must retain resident slots, including missing/invalid routes.
        for(int partition:{0,1}) {
            auto resident_addresses=addresses,streamed_addresses=addresses;
            for(std::size_t expert=0;expert<addresses.size();++expert) {
                if(int(expert%2)==partition)streamed_addresses[expert]=0;
                else resident_addresses[expert]=0;
            }
            check(cudaMemcpy(device_addresses.data,resident_addresses.data(),addresses.size()*sizeof(addresses[0]),cudaMemcpyHostToDevice));
            mfq::cuda::gguf::grouped_routed_matvec_q8(type,device_addresses.data,stride,addresses.size(),rows,columns,
                selected_device.data,groups.data,tokens,routes,quantized.data,routed,dy.data,nullptr);
            check(cudaMemcpy(device_addresses.data,streamed_addresses.data(),addresses.size()*sizeof(addresses[0]),cudaMemcpyHostToDevice));
            mfq::cuda::gguf::grouped_routed_matvec_q8(type,device_addresses.data,stride,addresses.size(),rows,columns,
                selected_device.data,groups.data,tokens,routes,quantized.data,routed,dy.data,nullptr,true);
            std::vector<float> split_output(y.size());
            check(cudaMemcpy(split_output.data(),dy.data,split_output.size()*sizeof(float),cudaMemcpyDeviceToHost));
            if(std::memcmp(split_output.data(),grouped_output.data(),split_output.size()*sizeof(float)))
                throw std::runtime_error(std::string(ggml_type_name(type))+" resident/streamed partition changed output bits");
        }
        check(cudaMemcpy(device_addresses.data,addresses.data(),addresses.size()*sizeof(addresses[0]),cudaMemcpyHostToDevice));
        for (int slot = 0; slot < tokens * routes; ++slot) for (int row = 0; row < rows; ++row) {
            const auto expert = selected[slot];
            double expected = 0., magnitude = 0.;
            if (expert >= 0 && expert < 3) for (int k = 0; k < columns; ++k) {
                const double product = double(reference[((row + expert) % rows) * columns + k]) *
                    x[(routed ? slot : slot / routes) * columns + k];
                expected += product; magnitude += std::abs(product);
            }
            if(!std::isfinite(grouped_output[slot*rows+row]) ||
                    std::abs(y[slot*rows+row]-grouped_output[slot*rows+row])>2e-6*magnitude+1e-6)
                throw std::runtime_error(std::string(ggml_type_name(type))+" grouped routed accumulation differs from the flat kernel");
            if (!std::isfinite(y[slot * rows + row]) ||
                    std::abs(y[slot * rows + row] - expected) > .008 * magnitude + 1e-6)
                throw std::runtime_error(std::string(ggml_type_name(type)) + " routed projection mismatch");
        }
    }
    if (mfq::cuda::gguf::grouped_f16_available(0)) for (bool routed : {false, true}) {
        constexpr int tokens=35, routes=2, count=tokens*routes;
        std::vector<std::int64_t> slots;
        std::vector<mfq::cuda::gguf::GemmJob> jobs(3);
        std::int64_t offsets[4]{};
        for(int expert=0;expert<3;++expert) {
            offsets[expert]=slots.size();
            for(int slot=0;slot<count;++slot) if((slot%5==0?2:0)==expert) slots.push_back(slot);
        }
        offsets[3]=slots.size();
        DeviceArray<std::int64_t> plan(slots.size());
        check(cudaMemcpy(plan.data,slots.data(),slots.size()*sizeof(slots[0]),cudaMemcpyHostToDevice));
        for(int expert=0;expert<3;++expert)jobs[expert]={device_banks.data+expert*bytes.size(),plan.data+offsets[expert],offsets[expert+1]-offsets[expert]};
        DeviceArray<mfq::cuda::gguf::GemmJob> gpu_jobs(jobs.size());
        check(cudaMemcpy(gpu_jobs.data,jobs.data(),jobs.size()*sizeof(jobs[0]),cudaMemcpyHostToDevice));
        std::vector<__half> x((routed?count:tokens)*columns), y(count*rows);
        for(std::size_t i=0;i<x.size();++i)x[i]=__float2half(std::cos(float(i)*.17f)*.1f);
        DeviceArray<__half> dx(x.size()),dy(y.size());
        check(cudaMemcpy(dx.data,x.data(),x.size()*sizeof(x[0]),cudaMemcpyHostToDevice));
        DeviceArray<__half> decoded(3*rows*columns),gathered(3*56*columns),scattered(count*columns);
        mfq::cuda::gguf::grouped_decode_f16(type,gpu_jobs.data,3,stride,rows,columns,decoded.data,nullptr);
        std::vector<__half> decoded_host(3*rows*columns);
        check(cudaMemcpy(decoded_host.data(),decoded.data,decoded_host.size()*sizeof(__half),cudaMemcpyDeviceToHost));
        for(int expert=0;expert<3;++expert)for(int row=0;row<rows;++row)for(int k=0;k<columns;++k) {
            const float expected=__half2float(__float2half(reference[((row+expert)%rows)*columns+k]));
            if(std::abs(__half2float(decoded_host[(expert*rows+row)*columns+k])-expected)>std::abs(expected)*1e-3f+1e-6f)
                throw std::runtime_error(std::string(ggml_type_name(type))+" grouped weight decode mismatch");
        }
        DeviceArray<__half> joined_decoded(6*rows*columns);
        check(cudaMemset(joined_decoded.data,0,6*rows*columns*sizeof(__half)));
        mfq::cuda::gguf::grouped_decode_f16(type,gpu_jobs.data,3,stride,rows,columns,
            joined_decoded.data+rows*columns,nullptr,2*rows*columns);
        std::vector<__half> joined_host(6*rows*columns);
        check(cudaMemcpy(joined_host.data(),joined_decoded.data,joined_host.size()*sizeof(__half),cudaMemcpyDeviceToHost));
        for(int expert=0;expert<3;++expert)for(int i=0;i<rows*columns;++i) {
            const auto at=expert*2*rows*columns+i;
            if(__half2float(joined_host[at])!=0 ||
                    std::memcmp(&joined_host[at+rows*columns],&decoded_host[expert*rows*columns+i],sizeof(__half)))
                throw std::runtime_error("GGUF joined decode changed the batch stride or another projection");
        }
        mfq::cuda::gguf::grouped_gather_f16(gpu_jobs.data,3,56,columns,dx.data,routed?1:routes,gathered.data,nullptr);
        std::vector<__half> gathered_host(3*56*columns);
        check(cudaMemcpy(gathered_host.data(),gathered.data,gathered_host.size()*sizeof(__half),cudaMemcpyDeviceToHost));
        for(int expert=0;expert<3;++expert)for(int row=0;row<56;++row)for(int k=0;k<columns;++k) {
            const float expected=row<jobs[expert].count?
                __half2float(x[(slots[offsets[expert]+row]/(routed?1:routes))*columns+k]):0.f;
            if(__half2float(gathered_host[(expert*56+row)*columns+k])!=expected)
                throw std::runtime_error("GGUF grouped gather changed rows or its zero padding");
        }
        mfq::cuda::gguf::grouped_scatter_f16(gpu_jobs.data,3,56,columns,gathered.data,scattered.data,nullptr);
        std::vector<__half> scattered_host(count*columns);
        check(cudaMemcpy(scattered_host.data(),scattered.data,scattered_host.size()*sizeof(__half),cudaMemcpyDeviceToHost));
        for(int row=0;row<count;++row)for(int k=0;k<columns;++k)
            if(__half2float(scattered_host[row*columns+k])!=__half2float(x[(row/(routed?1:routes))*columns+k]))
                throw std::runtime_error("GGUF grouped scatter changed row order");
        mfq::cuda::gguf::grouped_matmul_f16(type,gpu_jobs.data,jobs.size(),56,stride,rows,columns,
            dx.data,routes,routed,dy.data,nullptr);
        check(cudaMemcpy(y.data(),dy.data,y.size()*sizeof(y[0]),cudaMemcpyDeviceToHost));
        for(int slot=0;slot<count;++slot)for(int row=0;row<rows;++row) {
            const int expert=slot%5==0?2:0;
            double expected=0,magnitude=0;
            for(int k=0;k<columns;++k) {
                const double product=double(__half2float(__float2half(reference[((row+expert)%rows)*columns+k])))*
                    __half2float(x[(routed?slot:slot/routes)*columns+k]);
                expected+=product;magnitude+=std::abs(product);
            }
            if(!std::isfinite(__half2float(y[slot*rows+row])) ||
                std::abs(__half2float(y[slot*rows+row])-expected)>1e-3*magnitude+1e-5)
                throw std::runtime_error(std::string(ggml_type_name(type))+" grouped Tensor Core projection mismatch");
        }
    }
    std::printf("%s K=%d decode/embedding/graph/routed-M1,2,4/grouped-GEMM passed\n", ggml_type_name(type), columns);
    std::fflush(stdout);
}

void check_shared_swiglu_q8() {
    namespace g=mfq::cuda::gguf;
    for(int tokens:{1,2,3,4,8,32})for(int width:{32,96,640}) {
        std::vector<half> source(2*tokens*width);
        for(std::size_t i=0;i<source.size();++i)source[i]=__float2half_rn(std::sin(float(i)*.173f)*4.f);
        DeviceArray<half> input(source.size()),hidden(tokens*width);
        check(cudaMemcpy(input.data,source.data(),source.size()*sizeof(half),cudaMemcpyHostToDevice));
        for(bool interleaved:{false,true}) {
            if(interleaved && (tokens<2 || tokens>4))continue;
            const auto bytes=interleaved?g::interleaved_activation_bytes(tokens,width):tokens*g::q8_row_bytes(width);
            DeviceArray<uint8_t> actual(bytes+32),expected(bytes);
            check(cudaMemset(actual.data,0xcd,bytes+32));
            const auto reference=[&] {
                g::gate_up_swiglu_f16(input.data,tokens,width,hidden.data,0.f,nullptr);
                if(interleaved)g::quantize_input_interleaved(hidden.data,true,tokens,width,expected.data,nullptr,nullptr);
                else g::quantize_input_f16(hidden.data,tokens,width,expected.data,nullptr);
            };
            reference();g::gate_up_swiglu_q8_f16(input.data,tokens,width,actual.data+16,interleaved,nullptr);
            cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t replay=nullptr;
            check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            g::gate_up_swiglu_q8_f16(input.data,tokens,width,actual.data+16,interleaved,stream);
            check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&replay,graph,0));
            for(int change:{0,1,2}) {
                if(change) {
                    for(std::size_t i=0;i<source.size();++i)source[i]=__float2half_rn(std::sin(float(i)*(.17f+change*.04f))*.3f);
                    check(cudaMemcpy(input.data,source.data(),source.size()*sizeof(half),cudaMemcpyHostToDevice));reference();
                }
                check(cudaGraphLaunch(replay,stream));check(cudaStreamSynchronize(stream));
                std::vector<uint8_t> a(bytes+32),b(bytes);
                check(cudaMemcpy(a.data(),actual.data,a.size(),cudaMemcpyDeviceToHost));
                check(cudaMemcpy(b.data(),expected.data,b.size(),cudaMemcpyDeviceToHost));
                if(std::memcmp(a.data()+16,b.data(),bytes))throw std::runtime_error("shared SwiGLU Q8 changed packet bytes");
                for(int i=0;i<16;++i)if(a[i]!=0xcd || a[bytes+16+i]!=0xcd)
                    throw std::runtime_error("shared SwiGLU Q8 overwrote its output guard");
            }
            check(cudaGraphExecDestroy(replay));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
        }
    }
    std::puts("shared SwiGLU Q8 native/position-major/tails/changing graph/guards PASS");
}

void check_dense_history_tiles() {
    namespace g=mfq::cuda::gguf;
    int cases=0;double peak_error=0;
    const auto check_format=[&](ggml_type type) {
        constexpr int rows=7,guard=8;
        for(int columns:{int(3*ggml_blck_size(type)),3072}) {
            const auto stride=mfq::gguf::row_bytes(type,columns);
            std::vector<std::uint8_t> bytes(rows*stride);
            for(std::size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t((i*29+i/7+17)&63);
            std::vector<float> decoded(rows*columns);
            for(int row=0;row<rows;++row) {
                if(type==GGML_TYPE_Q8_K)dequantize_row_q8_K(
                    reinterpret_cast<const block_q8_K*>(bytes.data()+row*stride),decoded.data()+row*columns,columns);
                else ggml_get_type_traits(type)->to_float(bytes.data()+row*stride,decoded.data()+row*columns,columns);
            }
            DeviceArray<std::uint8_t> packed(bytes.size());
            check(cudaMemcpy(packed.data,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
            for(int tokens:{9,12,16,17,31,32,33}) {
                const auto count=tokens*rows;
                std::vector<float> x(tokens*columns),actual(count+2*guard),expected(count);
                DeviceArray<float> input(x.size()),out(actual.size()),reference(count);
                DeviceArray<ggml_fp16_t> half_out(count+2*guard);
                DeviceArray<std::uint8_t> activation(tokens*g::q8_row_bytes(columns));
                for(int replay=0;replay<2;++replay) {
                    for(std::size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.173f+replay*1.37f)*.1f;
                    check(cudaMemcpy(input.data,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
                    check(cudaMemset(out.data,0xcd,actual.size()*sizeof(float)));
                    check(cudaMemset(half_out.data,0xcd,(count+2*guard)*sizeof(ggml_fp16_t)));
                    g::quantize_input(input.data,tokens,columns,activation.data,nullptr);
                    g::matvec_q8(type,packed.data,stride,rows,columns,activation.data,tokens,out.data+guard,nullptr);
                    g::matvec_q8_f16(type,packed.data,stride,rows,columns,activation.data,tokens,half_out.data+guard,nullptr);
                    // The established M1 kernel gives an independent row/output
                    // stride reference, including the final incomplete tile.
                    for(int row=0;row<tokens;++row)
                        g::matvec_q8(type,packed.data,stride,rows,columns,
                            activation.data+row*g::q8_row_bytes(columns),1,reference.data+row*rows,nullptr);
                    check(cudaMemcpy(actual.data(),out.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(expected.data(),reference.data,expected.size()*sizeof(float),cudaMemcpyDeviceToHost));
                    std::vector<block_q8_1> packets(tokens*columns/32);
                    check(cudaMemcpy(packets.data(),activation.data,packets.size()*sizeof(packets[0]),cudaMemcpyDeviceToHost));
                    std::vector<ggml_fp16_t> half(count+2*guard);
                    check(cudaMemcpy(half.data(),half_out.data,half.size()*sizeof(half[0]),cudaMemcpyDeviceToHost));
                    for(int i=0;i<guard;++i) {
                        uint32_t before=0,after=0;
                        std::memcpy(&before,&actual[i],4);std::memcpy(&after,&actual[count+guard+i],4);
                        if(before!=0xcdcdcdcd || after!=0xcdcdcdcd || half[i]!=0xcdcd || half[count+guard+i]!=0xcdcd)
                            throw std::runtime_error("dense history tile overwrote output guard");
                    }
                    for(int i=0;i<count;++i) {
                        const double error=std::abs(double(actual[i+guard])-expected[i]);
                        peak_error=std::max(peak_error,error);
                        double magnitude=0;
                        for(int column=0;column<columns;++column) {
                            const auto& packet=packets[(i/rows)*(columns/32)+column/32];
                            magnitude+=std::abs(double(decoded[(i%rows)*columns+column])*
                                ggml_fp16_to_fp32(packet.d)*packet.qs[column%32]);
                        }
                        // Each established projection is checked against
                        // decoded products at 2e-6 * sum(abs(products)). Their
                        // different FP32 reduction trees can add both bounds.
                        if(!std::isfinite(actual[i+guard]) || !std::isfinite(expected[i]) ||
                                error>4e-6*magnitude+2e-6) {
                            std::fprintf(stderr,"dense history mismatch format=%s N=%d K=%d M=%d replay=%d index=%d actual=%g expected=%g error=%g\n",
                                ggml_type_name(type),rows,columns,tokens,replay,i,actual[i+guard],expected[i],error);
                            throw std::runtime_error(std::string(ggml_type_name(type))+" dense history tile differs from independent M1 projection");
                        }
                        if(half[i+guard]!=ggml_fp32_to_fp16(actual[i+guard]))
                            throw std::runtime_error("dense history tile changed fused half rounding");
                    }
                    ++cases;
                }
            }
        }
    };
#define TYLOINFER_GGUF_HISTORY_TEST(name) check_format(GGML_TYPE_##name);
    TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_HISTORY_TEST)
#undef TYLOINFER_GGUF_HISTORY_TEST
    std::printf("dense history all formats M9/12/16/17/31/32/33 short/long K odd N FP32/FP16 guards/changing input cases=%d max_abs=%g PASS\n",cases,peak_error);
    std::fflush(stdout);
}

void bench_routed(int skip_cases,bool joined=false,int only_tokens=0,int hidden_mode=-1,const char* only_format=nullptr,bool full_code_bits=false) {
    namespace g=mfq::cuda::gguf;
    constexpr int experts=24,routes=10,repetitions=64;
    int case_index=0;
    bool matched_format=false;
    for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ2_S,GGML_TYPE_Q2_0,
            GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ4_NL})
    for(bool down:{false,true}) {
        if(only_format && std::strcmp(only_format,ggml_type_name(type))!=0)continue;
        matched_format=true;
        if(joined && down)continue;
        // IQ4_NL has a routed single projection, but no paired Gate/Up API.
        if(joined && type==GGML_TYPE_IQ4_NL)continue;
        const int columns=down?(ggml_blck_size(type)>64?768:640):2560,rows=down?2560:640;
        const auto stride=mfq::gguf::row_bytes(type,columns);
        std::vector<uint8_t> image(experts*rows*stride);
        for(size_t i=0;i<image.size();++i)image[i]=uint8_t((i*29+i/11+7)&63);
        if(full_code_bits && (type==GGML_TYPE_IQ3_XXS || type==GGML_TYPE_IQ3_S ||
                type==GGML_TYPE_IQ2_XXS || type==GGML_TYPE_IQ2_XS || type==GGML_TYPE_IQ2_S)) {
            // Full sign/grid bits, with a finite block scale, as in check_type.
            for(size_t i=0;i<image.size();++i)image[i]=uint8_t(i*29+i/11+7);
            const uint16_t scale=0x3000;
            for(size_t i=0;i<image.size();i+=ggml_type_size(type))
                std::memcpy(image.data()+i,&scale,sizeof(scale));
        }
        DeviceArray<uint8_t> bank(image.size()),up_bank(joined?image.size():1);
        if(joined)check(cudaMemcpy(up_bank.data,image.data(),image.size(),cudaMemcpyHostToDevice));
        check(cudaMemcpy(bank.data,image.data(),image.size(),cudaMemcpyHostToDevice));
        std::vector<uint64_t> pointers(experts);
        for(int e=0;e<experts;++e)pointers[e]=e%5?reinterpret_cast<uint64_t>(bank.data+e*rows*stride):0;
        DeviceArray<uint64_t> addresses(experts),up_addresses(experts);
        if(joined){auto up_ptr=pointers;for(int e=0;e<experts;++e)if(up_ptr[e])up_ptr[e]=reinterpret_cast<uint64_t>(up_bank.data+e*rows*stride);
            check(cudaMemcpy(up_addresses.data,up_ptr.data(),experts*sizeof(uint64_t),cudaMemcpyHostToDevice));}
        check(cudaMemcpy(addresses.data,pointers.data(),experts*sizeof(uint64_t),cudaMemcpyHostToDevice));
        const std::vector<int> token_cases=only_tokens>4?std::vector<int>{only_tokens}:std::vector<int>{1,2,3,4};
        for(int diversity:{0,3})for(int tokens:token_cases) {
            if(case_index++<skip_cases || (only_tokens>0 && tokens!=only_tokens) || (only_tokens<0 && tokens==1))continue;
            const int count=tokens*routes,input_rows=down?count:tokens;
            std::vector<float> x(input_rows*columns);
            for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.17f)*.1f;
            DeviceArray<float> dx(x.size()),output(count*rows),up_output(count*rows),reference(count*rows);
            DeviceArray<uint8_t> hidden(count*g::q8_row_bytes(rows)),hidden_reference(count*g::q8_row_bytes(rows));
            check(cudaMemcpy(dx.data,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
            DeviceArray<uint8_t> qx(input_rows*g::q8_row_bytes(columns));
            g::quantize_input(dx.data,input_rows,columns,qx.data,nullptr);
            std::vector<int64_t> selected(count);
            for(int t=0;t<tokens;++t)for(int k=0;k<routes;++k)selected[t*routes+k]=(k+t*diversity)%experts;
            DeviceArray<int64_t> ids(count);
            check(cudaMemcpy(ids.data,selected.data(),count*sizeof(int64_t),cudaMemcpyHostToDevice));
            DeviceArray<g::RouteGroup> groups(count+1);
            g::prepare_route_groups(ids.data,count,tokens,groups.data,nullptr);
            g::routed_matvec_q8(type,addresses.data,stride,experts,rows,columns,ids.data,tokens,routes,
                qx.data,down,reference.data,nullptr);
            if(hidden_mode>=0) {
                if(!g::try_gate_up_matvec_q8(type,addresses.data,up_addresses.data,stride,stride,experts,rows,columns,
                        ids.data,groups.data,tokens,routes,qx.data,output.data,up_output.data,nullptr))
                    throw std::runtime_error("hidden benchmark reference pair kernel missing");
                g::quantize_moe_hidden(output.data,up_output.data,count,rows,true,0.f,hidden_reference.data,nullptr);
            }
            const auto execute=[&](cudaStream_t stream){
                if(hidden_mode==1) {
                    if(!g::try_gate_up_hidden_q8(type,addresses.data,up_addresses.data,stride,stride,experts,rows,columns,
                            ids.data,groups.data,tokens,routes,qx.data,hidden.data,true,0.f,stream))
                        throw std::runtime_error("hidden benchmark fused quantizer missing");
                }else if(joined){if(!g::try_gate_up_matvec_q8(type,addresses.data,up_addresses.data,stride,stride,experts,rows,columns,
                    ids.data,groups.data,tokens,routes,qx.data,output.data,up_output.data,stream))
                        throw std::runtime_error("fused Gate/Up benchmark did not use requested kernel");}
                else g::grouped_routed_matvec_q8(type,addresses.data,stride,experts,rows,columns,ids.data,groups.data,
                    tokens,routes,qx.data,down,output.data,stream);
                if(hidden_mode==0)g::quantize_moe_hidden(output.data,up_output.data,count,rows,true,0.f,hidden.data,stream);
            };
            execute(nullptr);
            std::vector<float> expected(count*rows),actual(expected.size());
            check(cudaMemcpy(expected.data(),reference.data,expected.size()*sizeof(float),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(actual.data(),output.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
            const bool bitwise=std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float))==0;
            if(!bitwise) {
                double magnitude=0,error=0,maximum=0,peak_error=0;
                for(std::size_t i=0;i<actual.size();++i) {
                    if(!std::isfinite(expected[i]) || !std::isfinite(actual[i]))
                        throw std::runtime_error("grouped routed benchmark produced nonfinite output");
                    const double delta=double(actual[i])-expected[i];
                    magnitude+=double(expected[i])*expected[i];error+=delta*delta;
                    maximum=std::max(maximum,std::abs(double(expected[i])));peak_error=std::max(peak_error,std::abs(delta));
                }
                if(error>1e-10*magnitude || peak_error>1e-4*maximum)
                    throw std::runtime_error("grouped routed benchmark exceeded numerical tolerance");
            }
            if(hidden_mode>=0) {
                std::vector<uint8_t> actual_hidden(count*g::q8_row_bytes(rows)),expected_hidden(actual_hidden.size());
                check(cudaMemcpy(actual_hidden.data(),hidden.data,actual_hidden.size(),cudaMemcpyDeviceToHost));
                check(cudaMemcpy(expected_hidden.data(),hidden_reference.data,actual_hidden.size(),cudaMemcpyDeviceToHost));
                if(actual_hidden!=expected_hidden)throw std::runtime_error("hidden benchmark Q8 bytes differ");
            }
            cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
            check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            for(int i=0;i<repetitions;++i)execute(stream);
            check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,0));
            for(int warm=0;warm<8;++warm)check(cudaGraphLaunch(exec,stream));
            check(cudaStreamSynchronize(stream));
            cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
            std::vector<float> samples(7);
            for(auto& sample:samples) {
                check(cudaEventRecord(begin,stream));check(cudaGraphLaunch(exec,stream));check(cudaEventRecord(end,stream));
                check(cudaEventSynchronize(end));check(cudaEventElapsedTime(&sample,begin,end));
            }
            std::sort(samples.begin(),samples.end());const float ms=samples[samples.size()/2];
            std::printf("%s format=%s N=%d K=%d M=%d diversity=%d us=%.6f bitwise=%d trials=7 median=1\n",
                hidden_mode==1?"hidden_fused_bench":hidden_mode==0?"hidden_separate_bench":joined?"gate_up_bench":"routed_bench",ggml_type_name(type),rows,columns,tokens,diversity,1000.*ms/repetitions,int(bitwise));
            check(cudaEventDestroy(begin));check(cudaEventDestroy(end));check(cudaGraphExecDestroy(exec));
            check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
        }
    }
    if(!matched_format)throw std::invalid_argument("Unknown routed benchmark format");
}

void bench_interleaved_gate_up(int tokens,int rows,int columns,int experts,int routes) {
    namespace g=mfq::cuda::gguf;
    if(tokens<1 || tokens>8 || rows<1 || columns<1 || columns%256 || experts<routes || routes<1)
        throw std::invalid_argument("interleaved Gate/Up benchmark geometry");
    struct Bank {
        ggml_type type;
        std::uint64_t stride;
        DeviceArray<std::uint8_t> gate,up;
        DeviceArray<std::uint64_t> gate_addresses,up_addresses;
        Bank(ggml_type t,int n,int k,int e):type(t),stride(mfq::gguf::row_bytes(t,k)),
            gate(std::size_t(e)*n*stride),up(std::size_t(e)*n*stride),gate_addresses(e),up_addresses(e) {
            std::vector<std::uint8_t> image(std::size_t(e)*n*stride);
            for(std::size_t i=0;i<image.size();++i)image[i]=std::uint8_t(i*29+i/11+7);
            const std::uint16_t scale=0x3000;
            for(std::size_t i=0;i<image.size();i+=ggml_type_size(t))std::memcpy(image.data()+i,&scale,sizeof(scale));
            check(cudaMemcpy(gate.data,image.data(),image.size(),cudaMemcpyHostToDevice));
            check(cudaMemcpy(up.data,image.data(),image.size(),cudaMemcpyHostToDevice));
            std::vector<std::uint64_t> pointers(e);
            for(int i=0;i<e;++i)pointers[i]=reinterpret_cast<std::uint64_t>(gate.data+i*n*stride);
            check(cudaMemcpy(gate_addresses.data,pointers.data(),e*sizeof(pointers[0]),cudaMemcpyHostToDevice));
            for(int i=0;i<e;++i)pointers[i]=reinterpret_cast<std::uint64_t>(up.data+i*n*stride);
            check(cudaMemcpy(up_addresses.data,pointers.data(),e*sizeof(pointers[0]),cudaMemcpyHostToDevice));
        }
    };
    std::vector<std::unique_ptr<Bank>> banks;
    for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S})
        banks.push_back(std::make_unique<Bank>(type,rows,columns,experts));
    std::vector<float> input(std::size_t(tokens)*columns);
    for(std::size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*.17f)*.1f;
    DeviceArray<float> x(input.size()),gate(std::size_t(tokens)*routes*rows),up(std::size_t(tokens)*routes*rows),reference(std::size_t(tokens)*routes*rows);
    DeviceArray<std::uint8_t> qx(tokens*g::q8_row_bytes(columns));
    check(cudaMemcpy(x.data,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));
    g::quantize_input(x.data,tokens,columns,qx.data,nullptr);
    DeviceArray<std::int64_t> ids(tokens*routes);
    DeviceArray<g::RouteGroup> groups(tokens*routes+1);
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(int diversity:{0,3}) {
        std::vector<std::int64_t> selected(tokens*routes);
        for(int t=0;t<tokens;++t)for(int k=0;k<routes;++k)selected[t*routes+k]=(k+t*diversity)%experts;
        check(cudaMemcpy(ids.data,selected.data(),selected.size()*sizeof(selected[0]),cudaMemcpyHostToDevice));
        g::prepare_route_groups(ids.data,tokens*routes,tokens,groups.data,nullptr);
        const auto execute=[&](Bank& bank,cudaStream_t selected_stream) {
            if(!g::try_gate_up_matvec_q8(bank.type,bank.gate_addresses.data,bank.up_addresses.data,
                    bank.stride,bank.stride,experts,rows,columns,ids.data,groups.data,tokens,routes,
                    qx.data,gate.data,up.data,selected_stream))
                throw std::runtime_error("interleaved Gate/Up dispatch missing");
        };
        for(const auto& bank:banks) {
            execute(*bank,nullptr);
            std::vector<float> actual(std::size_t(tokens)*routes*rows),expected(actual.size());
            for(bool is_up:{false,true}) {
                g::routed_matvec_q8(bank->type,is_up?bank->up_addresses.data:bank->gate_addresses.data,
                    bank->stride,experts,rows,columns,ids.data,tokens,routes,qx.data,false,reference.data,nullptr);
                check(cudaMemcpy(actual.data(),is_up?up.data:gate.data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
                check(cudaMemcpy(expected.data(),reference.data,expected.size()*sizeof(float),cudaMemcpyDeviceToHost));
                if(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)))
                    throw std::runtime_error("interleaved Gate/Up changed reference bits");
            }
        }
        constexpr int repeats=32;
        const auto measure=[&](int single) {
            cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            for(int i=0;i<repeats;++i) {
                if(single>=0)execute(*banks[single],stream);
                else for(const auto& bank:banks)execute(*bank,stream);
            }
            check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
            for(int i=0;i<8;++i)check(cudaGraphLaunch(executable,stream));
            cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
            std::vector<float> samples(7);
            for(auto& sample:samples) {
                check(cudaEventRecord(begin,stream));check(cudaGraphLaunch(executable,stream));
                check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));check(cudaEventElapsedTime(&sample,begin,end));
            }
            std::sort(samples.begin(),samples.end());
            check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
            check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
            return 1000.*samples[samples.size()/2]/repeats;
        };
        double isolated=0;
        for(int i=0;i<int(banks.size());++i){const auto us=measure(i);isolated+=us;
            std::printf("interleave_isolated format=%s M=%d diversity=%d us=%.6f\n",ggml_type_name(banks[i]->type),tokens,diversity,us);}
        const auto mixed=measure(-1);
        std::printf("interleave_gate_up N=%d K=%d M=%d experts=%d routes=%d diversity=%d isolated_sum_us=%.6f mixed_sum_us=%.6f ratio=%.6f reference_bitwise=1\n",
            rows,columns,tokens,experts,routes,diversity,isolated,mixed,mixed/isolated);
    }
    check(cudaStreamDestroy(stream));
}

int main(int argc,char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        if(argc==2 && std::strcmp(argv[1],"--compact-shared-check")==0) {
            // Odd K-block counts and row tails exercise the cooperative
            // original-codebook load, absent experts and grouped M=1..8.
            for(int type=GGML_TYPE_TQ2_T;type<=GGML_TYPE_TQK4;++type)
                check_type(ggml_type(type),2176,17);
            // More output tiles bound grid.y, so CTAs reuse the table while
            // advancing between route groups and writing activation scratch.
            check_type(GGML_TYPE_TQK6,2176,257);
            std::puts("Compact8 shared codebook / tails / missing experts / persistent groups PASS");
            std::fflush(stdout);return 0;
        }
        if(argc==2 && std::strcmp(argv[1],"--iq3-s-tile-check")==0) {
            for(int rows:{5,32})for(int columns:{256,768,2816})
                check_type(GGML_TYPE_IQ3_S,columns,rows);
            std::puts("IQ3_S paired tile short/long K odd/quantized N PASS");
            std::fflush(stdout);return 0;
        }
        if(argc==7 && std::strcmp(argv[1],"--gate-up-interleave-bench")==0) {
            bench_interleaved_gate_up(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]));return 0;
        }
        if((argc==3 || argc==4) && std::strcmp(argv[1],"--gate-up-full-code-bench")==0) {
            std::puts("diagnostic_full_code_bits=1");
            bench_routed(0,true,std::stoi(argv[2]),-1,argc==4?argv[3]:nullptr,true);return 0;
        }
        if(argc==5 && std::strcmp(argv[1],"--gate-up-pressure-bench")==0) {
            const double fraction=std::stod(argv[3]);
            if(fraction<0. || fraction>=1.)throw std::invalid_argument("pressure fraction must be in [0,1)");
            std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
            const auto bytes=std::size_t(double(free)*fraction);
            DeviceArray<std::uint8_t> pressure(bytes?bytes:1);
            if(bytes)check(cudaMemset(pressure.data,1,bytes));
            check(cudaDeviceSynchronize());
            std::printf("diagnostic_pressure_bytes=%llu initial_free_bytes=%llu\n",
                static_cast<unsigned long long>(bytes),static_cast<unsigned long long>(free));
            bench_routed(0,true,std::stoi(argv[2]),-1,argv[4]);return 0;
        }
        if(argc>=3 && std::strcmp(argv[1],"--hidden-bench")==0){bench_routed(0,true,argc>3?std::stoi(argv[3]):0,std::stoi(argv[2]));return 0;}
        if(argc==2 && std::strcmp(argv[1],"--hidden-check")==0){check_type(GGML_TYPE_IQ3_S,256,32);return 0;}
        if(argc==2 && std::strcmp(argv[1],"--dense-history-check")==0){check_dense_history_tiles();return 0;}
        if(argc==2 && std::strcmp(argv[1],"--shared-hidden-check")==0){check_shared_swiglu_q8();return 0;}
        if(argc>=2 && std::strcmp(argv[1],"--gate-up-bench")==0){bench_routed(argc>3?std::stoi(argv[3]):0,true,argc>2?std::stoi(argv[2]):0,-1,argc>4?argv[4]:nullptr);return 0;}
        if(argc>=2 && std::strcmp(argv[1],"--routed-bench")==0){bench_routed(argc>2?std::stoi(argv[2]):0,false,argc>3?std::stoi(argv[3]):0,-1,argc>4?argv[4]:nullptr);return 0;}
        const std::vector<std::size_t> sizes{13,65539,1000000};
        std::vector<mfq::cuda::gguf::UploadChunk> chunks;
        std::vector<std::uint8_t> packet(3*sizeof(mfq::cuda::gguf::UploadChunk));
        DeviceArray<std::uint8_t> scattered(1200000);
        check(cudaMemset(scattered.data,255,1200000));
        std::size_t destination=16;
        for(const auto size:sizes) {
            const auto offset=(packet.size()+15)&~std::size_t(15);packet.resize(offset+size);
            for(std::size_t i=0;i<size;++i)packet[offset+i]=std::uint8_t(i*37+size);
            chunks.push_back({reinterpret_cast<std::uint64_t>(scattered.data+destination),offset,size,0});
            destination=(destination+size+31)&~std::size_t(15);
        }
        std::memcpy(packet.data(),chunks.data(),chunks.size()*sizeof(chunks[0]));
        DeviceArray<std::uint8_t> gpu_packet(packet.size());
        check(cudaMemcpy(gpu_packet.data,packet.data(),packet.size(),cudaMemcpyHostToDevice));
        mfq::cuda::gguf::scatter_upload(gpu_packet.data,chunks.size(),sizes.back(),nullptr);
        std::vector<std::uint8_t> actual(1200000),expected(actual.size(),255);
        for(const auto& chunk:chunks) {
            const auto offset=std::size_t(chunk.destination-reinterpret_cast<std::uint64_t>(scattered.data));
            std::memcpy(expected.data()+offset,packet.data()+chunk.offset,chunk.bytes);
        }
        check(cudaMemcpy(actual.data(),scattered.data,actual.size(),cudaMemcpyDeviceToHost));
        if(actual!=expected)throw std::runtime_error("GGUF upload scatter changed a tail or adjacent bytes");
        for (const int width : {256, 768}) {
#define TYLOINFER_GGUF_TEST(name) check_type(GGML_TYPE_##name, width);
            TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_TEST)
#undef TYLOINFER_GGUF_TEST
        }
        for(auto type:{GGML_TYPE_Q4_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL})check_type(type,96);
        // Odd N, long K and M=1..8 exercise cooperative dense reductions,
        // alongside the original single-token and short-axis paths.
        for(auto type:{GGML_TYPE_Q8_0,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ4_XS})
            check_type(type,3072);
        check_type(GGML_TYPE_IQ4_XS,6144);
        for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S})
            check_type(type,2816);
        for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,GGML_TYPE_IQ4_XS,GGML_TYPE_Q2_0})
            for(int width:{256,768,2816})check_type(type,width,32);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::puts("GGUF quant GPU suite PASS");
    std::fflush(stdout);
    return 0;
}
