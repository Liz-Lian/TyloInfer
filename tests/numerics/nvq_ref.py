"""Freeze the external MFQ NVQ/NPQ packed-input backward fixtures."""

from pathlib import Path

import numpy as np
import torch

from .case import Case, CaseWriter, save_tensor, write_result
from .mfq_source import file_hash, open_source
from .torch_env import device as resolve_device
from .torch_env import execution, scope, settings

_CASES = (
    'nvq1_l',
    'nvq2',
    'nvq3',
    'nvq2_exec',
    'nvq2j',
    'nvq2j_exec',
    'npq0_l',
    'nvq1_s',
    'npq0_s',
    'nvq3j',
    'nvq3j_analytic',
    'nvq3j_512',
    'nvq2j_1024',
    'nvq2j_4096',
    'nvq3j_1024',
)
_JSC_SPECS = {
    'nvq2j': 'NVQ2_E8',
    'nvq2j_exec': 'NVQ2_E8',
    'nvq2j_1024': 'NVQ2_E8_1024',
    'nvq2j_4096': 'NVQ2_E8_4096',
    'nvq3j': 'NVQ3_D4',
    'nvq3j_analytic': 'NVQ3_D4',
    'nvq3j_512': 'NVQ3_D4_512',
    'nvq3j_1024': 'NVQ3_D4_1024',
}
_FIELDS = (
    'neuron_scale',
    'sub_scale',
    'indices',
    'signs',
    'delta_sign',
    'codebook',
    'scale_lut',
    'bank_for_state',
    'state',
    'codebooks',
    'first_codebooks',
    'second_codebooks',
)


def _fixture(case, modules):
    """Reproduce one original unrounded quantized-domain CPU fixture."""
    nvq = modules['mfq.formats.nvq']
    if case in {'nvq1_l', 'nvq2', 'nvq3', 'nvq2_exec'}:
        format_id = {'nvq1_l': 1, 'nvq2': 2, 'nvq2_exec': 2, 'nvq3': 3}[case]
        rng = np.random.default_rng(20260716 + format_id)
        weight = rng.normal(0.0, 0.05, size=(24, 80)).astype(np.float32)
        if format_id == 1:
            fmt = modules['mfq.formats.nvq1_l']
            return modules['mfq.quantize.nvq1_l_quant'].quantize(
                weight,
                fmt.NVQ1_L_T8_S3,
                anchor_multipliers=(1.0,),
                refine_steps=1,
                group_chunk=64,
            )
        base = nvq.NVQ2_E8 if format_id == 2 else nvq.NVQ3_D4
        spec = nvq.NvqSpec(base.codebook, groupsize=24, sub_bits=4, sign_mode='even')
        return modules['mfq.quantize.nvq_quant'].quantize(
            weight, spec, search_steps=3, group_chunk=128
        )
    if case in _JSC_SPECS:
        spec = getattr(nvq, _JSC_SPECS[case])
        rng = np.random.default_rng(20260731)
        builtin = nvq.codebook_for(spec)
        bank0 = (builtin.astype(np.int16) * 8).astype(np.int8)
        bank1 = np.clip(
            np.roll(bank0, 7, axis=0).astype(np.int16) + rng.integers(-2, 3, size=bank0.shape),
            1,
            127,
        ).astype(np.int8)
        scale_lut = np.arange(16, dtype=np.float32)
        if case == 'nvq3j_analytic':
            scale_lut = np.arange(16, dtype=np.float32) // 2 + 1.0
        return nvq.NvqJscTensor(
            shape=(24, 80),
            axis=0,
            neuron_len=80,
            neuron_scale=rng.uniform(2e-5, 8e-5, size=24).astype(np.float32),
            scale_lut=scale_lut,
            bank_for_state=(np.arange(16, dtype=np.uint8) & 1),
            state=rng.integers(0, 16, size=(24, 4), dtype=np.uint8),
            indices=rng.integers(
                0,
                spec.codebook_entries,
                size=(24, (80 + spec.vector_size - 1) // spec.vector_size),
                dtype=np.uint8 if spec.index_bits <= 8 else np.uint16,
            ),
            signs=rng.integers(0, 128, size=(24, 10), dtype=np.uint8),
            codebooks=np.stack((bank0, bank1)),
            base_spec=spec,
        )
    if case == 'nvq1_s':
        fmt = modules['mfq.formats.nvq1_s']
        rng = np.random.default_rng(20260723 + 24 + 80)
        return fmt.Nvq1STensor(
            spec=fmt.NVQ1_S,
            shape=(24, 80),
            axis=0,
            neuron_len=80,
            neuron_scale=rng.uniform(2e-3, 8e-3, size=24).astype(np.float32),
            sub_scale=rng.integers(0, 16, size=(24, 4), dtype=np.uint8),
            indices=rng.integers(0, 512, size=(24, 10), dtype=np.uint16),
            delta_sign=rng.integers(0, 2, size=(24, 4), dtype=np.uint8),
            codebook=fmt.NVQ1_S_SYNTHETIC_BANKS.copy(),
        )
    if case not in {'npq0_l', 'npq0_s'}:
        raise ValueError(f'Unsupported NVQ fixture: {case}')
    large = case == 'npq0_l'
    rng = np.random.default_rng(20260722 if large else 20260724 + 24 + 80)
    states = 8 if large else 4
    first = rng.integers(-48, 49, size=(states, 8, 4), dtype=np.int16).astype(np.int8)
    second = rng.integers(-48, 49, size=(states, 16 if large else 8, 4), dtype=np.int16).astype(
        np.int8
    )
    fmt = modules['mfq.formats.npq0_l' if large else 'mfq.formats.npq0_s']
    cls = fmt.Npq0LTensor if large else fmt.Npq0STensor
    return cls(
        shape=(24, 80),
        axis=0,
        neuron_len=80,
        neuron_scale=rng.uniform(2e-4, 8e-4, size=24).astype(np.float16).astype(np.float32),
        scale_lut=np.linspace(0.125 if large else 0.25, 1.0, states, dtype=np.float16).astype(
            np.float32
        ),
        state=rng.integers(0, states, size=(24, 4), dtype=np.uint8),
        indices=rng.integers(0, 128 if large else 64, size=(24, 10), dtype=np.uint8),
        first_codebooks=first,
        second_codebooks=second,
    )


def _rebuild(case, modules):
    """Restore original logical components without rounding FP32 anchors."""
    label = case.data['config']['fixture']
    parts = {
        name: case.tensor('weight_' + name)
        for name in _FIELDS
        if 'weight_' + name in case.data['inputs']
    }
    common = {'shape': (24, 80), 'axis': 0, 'neuron_len': 80}
    if label == 'nvq1_l':
        fmt = modules['mfq.formats.nvq1_l']
        return fmt.Nvq1LTensor(spec=fmt.NVQ1_L_T8_S3, **common, **parts)
    if label == 'nvq1_s':
        fmt = modules['mfq.formats.nvq1_s']
        return fmt.Nvq1STensor(spec=fmt.NVQ1_S, **common, **parts)
    if label in {'nvq2', 'nvq3', 'nvq2_exec'}:
        fmt = modules['mfq.formats.nvq']
        base = fmt.NVQ3_D4 if label == 'nvq3' else fmt.NVQ2_E8
        return fmt.NvqTensor(
            spec=fmt.NvqSpec(base.codebook, groupsize=24, sub_bits=4, sign_mode='even'),
            **common,
            **parts,
        )
    if label in _JSC_SPECS:
        fmt = modules['mfq.formats.nvq']
        alias = parts.pop('sub_scale', None)
        if alias is None or 'state' not in parts or not np.array_equal(alias, parts['state']):
            raise ValueError('NVQ JSC sub-scale alias differs from its state')
        return fmt.NvqJscTensor(base_spec=getattr(fmt, _JSC_SPECS[label]), **common, **parts)
    if label in {'npq0_l', 'npq0_s'}:
        fmt = modules['mfq.formats.npq0_l' if label == 'npq0_l' else 'mfq.formats.npq0_s']
        return (fmt.Npq0LTensor if label == 'npq0_l' else fmt.Npq0STensor)(**common, **parts)
    raise ValueError(f'Unsupported NVQ fixture: {label}')


def _runtime(tensor, label, modules):
    """Build the source wrapper's actual packed buffers using CPU format encoders."""
    nvq = modules['mfq.formats.nvq']
    pack = nvq._pack_bits
    fmt_id = _CASES.index(label) + 1
    if label in {'npq0_l', 'npq0_s'}:
        large = label == 'npq0_l'
        fmt = modules['mfq.formats.npq0_l' if large else 'mfq.formats.npq0_s']
        tables = (fmt.pack_npq0_l_tables if large else fmt.pack_npq0_s_tables)(
            tensor.scale_lut, tensor.first_codebooks, tensor.second_codebooks
        )
        if large:
            codebook = np.frombuffer(tables, dtype=np.int8).copy()
        else:
            first = np.broadcast_to(tensor.first_codebooks[:, None, :, :], (4, 8, 8, 4))
            second = np.broadcast_to(tensor.second_codebooks[:, :, None, :], (4, 8, 8, 4))
            product = np.concatenate((first, second), axis=-1).reshape(4, 64, 8)
            codebook = np.concatenate(
                (np.frombuffer(tables, dtype=np.int8)[:64], product.reshape(-1))
            )
        indices = pack(tensor.indices, 7 if large else 6)
        aux = b''
        state = pack(tensor.state, 3 if large else 2)
    elif label in {'nvq1_l', 'nvq1_s'}:
        fmt = modules['mfq.formats.nvq1_l']
        book = fmt.IQ1S_TERNARY_2048 if tensor.codebook is None else tensor.codebook
        codebook = np.asarray(book, dtype=np.int8).reshape(-1, 8)
        indices = fmt._pack_bits(tensor.indices, tensor.spec.index_bits)
        aux = fmt._pack_bits(tensor.delta_sign, 1)
        state = fmt._pack_bits(tensor.sub_scale, tensor.spec.sub_bits)
    elif label in _JSC_SPECS:
        codebook = np.frombuffer(nvq.pack_jsc_metadata(tensor), dtype=np.int8).copy()
        indices = pack(tensor.indices, tensor.spec.index_bits)
        aux = pack(tensor.signs, 7)
        state = pack(tensor.state, 4)
    else:
        book = tensor.codebook if tensor.codebook is not None else nvq.codebook_for(tensor.spec)
        codebook = np.asarray(book, dtype=np.int8)
        indices = pack(tensor.indices, tensor.spec.index_bits)
        aux = pack(tensor.signs, 7)
        state = pack(tensor.sub_scale, tensor.spec.sub_bits)
    if label in {'nvq2_exec', 'nvq2j_exec'}:
        signs = np.asarray(tensor.signs, dtype=np.uint8)
        parity = np.fromiter((i.bit_count() & 1 for i in range(128)), dtype=np.uint8, count=128)
        paired = np.empty((*signs.shape, 2), dtype=np.uint8)
        paired[..., 0] = tensor.indices
        paired[..., 1] = signs | (parity[signs] << 7)
        indices, aux = paired.tobytes(), b''
    if label == 'nvq3j_analytic':
        # The source wrapper selects format 11 only for the analytic metadata.
        metadata = codebook.view(np.uint8)
        if metadata[3] != 1 or metadata[1] != 2:
            raise ValueError('Analytic JSC metadata does not select format 11')
    return {
        'indices_packed': np.frombuffer(indices, dtype=np.uint8).copy(),
        'aux_packed': np.frombuffer(aux, dtype=np.uint8).copy(),
        'sub_scale_packed': np.frombuffer(state, dtype=np.uint8).copy(),
        'codebook_packed': np.asarray(codebook, dtype=np.int8).copy(),
        'neuron_scale': np.asarray(tensor.neuron_scale, dtype=np.float32).copy(),
    }, fmt_id


def _decode(tensor, label, modules):
    """Use the source test's independent NumPy decoder for the tensor family."""
    if label in _JSC_SPECS:
        return modules['mfq.quantize.nvq_jsc'].dequantize_nvq_jsc(tensor)
    if label in {'nvq1_l', 'nvq1_s'}:
        name = 'mfq.quantize.nvq1_l_quant' if label == 'nvq1_l' else 'mfq.quantize.nvq1_s_quant'
        return modules[name].dequantize(tensor)
    if label in {'npq0_l', 'npq0_s'}:
        name = 'mfq.quantize.npq0_l' if label == 'npq0_l' else 'mfq.quantize.npq0_s'
        return getattr(modules[name], 'dequantize_' + label)(tensor)
    return modules['mfq.quantize.nvq_quant'].dequantize(tensor)


def _verified(case, source_root):
    """Verify the external source and saved runtime buffers against the fixed components."""
    if (
        case.data['op'] != 'nvq_backward_input'
        or case.data['source'].get('generator') != 'tests/numerics/nvq_ref.py'
    ):
        raise ValueError('Unsupported NVQ backward case')
    modules, identity = open_source(source_root)
    if identity != case.data['source']['mfq_source']:
        raise ValueError('MFQ source identity differs from the case')
    label = case.data['config']['fixture']
    tensor = _rebuild(case, modules)
    runtime, fmt_id = _runtime(tensor, label, modules)
    if fmt_id != case.data['config']['format']:
        raise ValueError('NVQ runtime format does not match the case')
    storage = {'nvq2_exec': 2, 'nvq2j_exec': 5}
    config = case.data['config']
    if label in storage:
        if (
            type(config.get('storage_format')) is not int
            or config['storage_format'] != storage[label]
        ):
            raise ValueError('NVQ storage format does not match the source wrapper')
    elif 'storage_format' in config:
        raise ValueError('Unexpected NVQ storage format for the source wrapper')
    for name, value in runtime.items():
        fixed = case.tensor(name)
        if value.dtype != fixed.dtype or not np.array_equal(value, fixed):
            raise ValueError(f'NVQ runtime buffer differs from fixed components: {name}')
    return modules, identity, tensor, runtime, label


def reference(case: Case, root: Path, source_root: Path, device: str = 'cuda') -> Path:
    """Compute both original FP16 gradient expectations from fixed CPU components."""
    modules, identity, tensor, _, label = _verified(case, source_root)
    device = resolve_device(device)
    with scope(case.data['config']['compute']):
        dense = torch.as_tensor(
            np.ascontiguousarray(_decode(tensor, label, modules)),
            device=device,
            dtype=torch.float16,
        )
        gradient = torch.from_numpy(case.tensor('output_gradient')).to(device=device)
        expected = (gradient @ dense).detach().contiguous().cpu().numpy()
        outputs = {
            name: save_tensor(root, name, expected)
            for name in ('input_gradient', 'autograd_input_gradient')
        }
        metadata = execution(
            device, 'NumPy CPU decode to float16; Torch float16 matrix multiplication'
        )
        metadata.update(
            generator_sha256=file_hash(__file__),
            mfq_source=identity,
            independence='CPU reference does not call CUDA kernels; MFQ format history is shared',
        )
        if open_source(source_root)[1] != identity:
            raise ValueError('MFQ source changed during reference calculation')
        return write_result(case, root, outputs, metadata)


def candidate(case: Case, root: Path, source_root: Path, device: str = 'cuda') -> Path:
    """Execute local packed NVQ backward and wrapper autograd on fixed inputs."""
    if case.data.get('actual_role') != 'candidate':
        raise ValueError('NVQ candidate requires the candidate result role')
    from tyloinfer.kernels.cuda import nvq_matmul as implementation
    from tyloinfer.kernels.cuda._ext import ext

    _, identity, tensor, runtime, _ = _verified(case, source_root)
    config = case.data['config']
    rows, group = config['rows'], config['groupsize']
    if (rows not in (1, 2, 4, 8, 16) or config['weight_shape'] != [24, 80]
            or group != tensor.spec.groupsize or config['sub_bits'] != tensor.spec.sub_bits
            or config['sign_mode'] != 0):
        raise ValueError('NVQ candidate parameters differ from the fixed components')
    device = resolve_device(device)
    with scope(config['compute']), torch.cuda.device(device):
        weight = {name: torch.from_numpy(case.tensor(name)).to(device)
                  for name in ('indices_packed', 'aux_packed', 'sub_scale_packed', 'neuron_scale')}
        weight['codebook'] = torch.from_numpy(case.tensor('codebook_packed')).to(device)
        weight.update(out=24, ng=(80 + group - 1) // group, gs=group, neuron_len=80,
                      sub_bits=config['sub_bits'], format=config['format'],
                      sign_mode=config['sign_mode'])
        if 'storage_format' in config:
            weight['storage_format'] = config['storage_format']
        gradient = torch.from_numpy(case.tensor('output_gradient')).to(device)
        source = torch.from_numpy(case.tensor('source')).to(device)
        if (gradient.dtype != torch.float16 or gradient.shape != (rows, 24)
                or source.dtype != torch.float16 or source.shape != (rows, 80)):
            raise ValueError('NVQ candidate inputs differ from the original case')
        direct = implementation.nvq_backward_input(weight, gradient)
        source.requires_grad_(True)
        (implementation.nvq_matmul(weight, source) * gradient).sum().backward()
        outputs = {'input_gradient': direct, 'autograd_input_gradient': source.grad}
        specs = {name: save_tensor(root, name, value.detach().contiguous().cpu().numpy())
                 for name, value in outputs.items()}
        module = ext()
        provenance = execution(device, 'Local NVQ packed backward and wrapper autograd')
        provenance.update(category='python_candidate', native_worker_executed=False,
                          forward_path=implementation._select_matmul_path(weight, rows),
                          mfq_source=identity, wrapper_sha256=file_hash(implementation.__file__),
                          jit_module=str(Path(module.__file__).resolve()),
                          jit_sha256=file_hash(module.__file__), generator_sha256=file_hash(__file__))
        if open_source(source_root)[1] != identity:
            raise ValueError('MFQ source changed during candidate execution')
        return write_result(case, root, specs, provenance, role='candidate')


def generate(root: Path, source_root: Path, device: str = 'cuda') -> list[Path]:
    """Freeze all 15 external NVQ/NPQ formats and five original row counts."""
    modules, identity = open_source(source_root)
    device = resolve_device(device)
    source = Path(source_root) / 'tests/test_nvq_kernels_cuda.py'
    results = []
    for format_id, label in enumerate(_CASES, 1):
        for rows in (1, 2, 4, 8, 16):
            tensor = _fixture(label, modules)
            runtime, selected = _runtime(tensor, label, modules)
            if selected != format_id:
                raise ValueError('Incorrect source runtime format')
            seed = 20261112 + format_id * 100 + rows
            torch.manual_seed(seed)
            gradient = torch.randn(rows, 24, device=device, dtype=torch.float16)
            source_input = torch.randn(rows, 80, device=device, dtype=torch.float16)
            writer = CaseWriter(
                root,
                f'nvq_backward_{label}_{rows}',
                'nvq_backward_input',
                {
                    'fixture': label,
                    'format': format_id,
                    'rows': rows,
                    'weight_shape': [24, 80],
                    'groupsize': tensor.spec.groupsize,
                    'sub_bits': tensor.spec.sub_bits,
                    'sign_mode': 0,
                    'compute': settings(),
                    **({'storage_format': {4: 2, 6: 5}[format_id]} if format_id in {4, 6} else {}),
                    'wrapper': 'to_gpu_nvq_exec'
                    if label in {'nvq2_exec', 'nvq2j_exec'}
                    else 'to_gpu_nvq',
                    'behavior_observations': {
                        'packed_backward': ['input_gradient'],
                        'autograd_input_gradient': ['autograd_input_gradient'],
                    },
                },
                {
                    'generator': 'tests/numerics/nvq_ref.py',
                    'generator_sha256': file_hash(__file__),
                    'test': 'tests/test_nvq_kernels_cuda.py::test_nvq_packed_backward_and_autograd_match_dequant',
                    'test_origin': 'external_mfq_checkout',
                    'test_sha256': file_hash(source),
                    'mfq_source': identity,
                    'input_seed': seed,
                    'seed_origin': 'chosen; original Torch input was unseeded',
                    'independence': 'external CPU decode versus CUDA execution; shared MFQ format history',
                },
                actual_role='candidate',
            )
            for field in _FIELDS:
                value = getattr(tensor, field, None)
                if value is not None:
                    writer.input('weight_' + field, value)
            for name, value in runtime.items():
                writer.input(name, value)
            writer.input('output_gradient', gradient.detach().cpu().numpy())
            writer.input('source', source_input.detach().cpu().numpy())
            rule = {'kind': 'close', 'atol': 0.004, 'rtol': 0.004}
            for name in ('input_gradient', 'autograd_input_gradient'):
                writer.output(name, (rows, 80), 'float16', rule)
            case = writer.finish()
            reference(case, case.root / 'reference', source_root, device)
            candidate(case, case.root / 'candidate', source_root, device)
            results.append(case.root / 'case.json')
    return results
