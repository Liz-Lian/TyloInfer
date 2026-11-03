#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace tyloinfer::qsa {

// Equal-length rows: predicted distortion is E * 4^-b. The marginal gain
// falls by four after each added bit, so greedy allocation is optimal for
// this energy proxy. No quantization candidates are evaluated here.
inline std::vector<std::uint8_t> allocate_energy_bits(const std::vector<double>& energy,int average_bits) {
    if(energy.empty() || average_bits<1 || average_bits>8 ||
       energy.size()>std::numeric_limits<std::size_t>::max()/8)
        throw std::invalid_argument("invalid QSA energy allocation budget");
    struct Entry {double gain;std::size_t row;};
    struct Less {
        bool operator()(const Entry& a,const Entry& b)const {
            return a.gain!=b.gain?a.gain<b.gain:a.row>b.row;
        }
    };
    std::priority_queue<Entry,std::vector<Entry>,Less> heap;
    std::vector<std::uint8_t> bits(energy.size(),1);
    for(std::size_t row=0;row<energy.size();++row) {
        if(!std::isfinite(energy[row]) || energy[row]<0)
            throw std::invalid_argument("invalid QSA row energy");
        heap.push({energy[row]*.1875,row});
    }
    for(std::size_t left=energy.size()*std::size_t(average_bits-1);left;--left) {
        auto best=heap.top();heap.pop();
        if(++bits[best.row]<8)heap.push({best.gain*.25,best.row});
    }
    return bits;
}
} // namespace tyloinfer::qsa
