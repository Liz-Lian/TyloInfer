"""Freeze released MFQ CPU fixture profiles using a separate source checkout."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .case import Case, CaseWriter, save_tensor, write_result
from .mfq_source import file_hash as _hash
from .mfq_source import open_source as _source


def _tensor(profile, rows, width, rng, modules):
    """Build the original random packed-domain weight for one released profile."""
    nint = modules['mfq.formats.nint']
    nvq = modules['mfq.formats.nvq']
    if profile.startswith('nint'):
        bits, groupsize, sub_bits = {
            'nint4': (4, 24, 6),
            'nint5': (5, 28, 7),
            'nint6': (6, 32, 8),
            'nint8': (8, 32, 8),
        }[profile]
        groups = math.ceil(width / groupsize)
        return nint.NintTensor(
            spec=nint.NintSpec(bits, groupsize, sub_bits),
            shape=(rows, width),
            axis=0,
            q=rng.integers(0, 1 << bits, (rows, groups, groupsize), dtype=np.uint8),
            neuron_scale=rng.uniform(0.002, 0.02, rows).astype(np.float16).astype(np.float32)
            / (1 << bits),
            neuron_min=rng.uniform(0.0001, 0.002, rows).astype(np.float16).astype(np.float32),
            sub_scale=rng.integers(0, 1 << sub_bits, (rows, groups), dtype=np.uint8),
            sub_min=rng.integers(0, 1 << sub_bits, (rows, groups), dtype=np.uint8),
            neuron_len=width,
        )
    if profile in {'nvq1-s', 'nvq1-l'}:
        small = profile == 'nvq1-s'
        cls = (
            modules['mfq.formats.nvq1_s'].Nvq1STensor
            if small
            else modules['mfq.formats.nvq1_l'].Nvq1LTensor
        )
        spec = (
            modules['mfq.formats.nvq1_s'].Nvq1SSpec()
            if small
            else modules['mfq.formats.nvq1_l'].NVQ1_L_T8_S4
        )
        groups = math.ceil(width / 24)
        return cls(
            spec=spec,
            shape=(rows, width),
            axis=0,
            neuron_len=width,
            neuron_scale=rng.uniform(0.002, 0.02, rows).astype(np.float16).astype(np.float32),
            sub_scale=rng.integers(0, 16, (rows, groups), dtype=np.uint8),
            indices=rng.integers(0, 512 if small else 2048, (rows, width // 8), dtype=np.uint16),
            delta_sign=rng.integers(0, 2, (rows, groups), dtype=np.uint8),
        )
    spec = getattr(
        nvq,
        {
            'nvq2j': 'NVQ2_E8',
            'nvq2j-l': 'NVQ2_E8_1024',
            'nvq2j-xl': 'NVQ2_E8_4096',
            'nvq3j': 'NVQ3_D4',
            'nvq3j-512': 'NVQ3_D4_512',
            'nvq3j-l': 'NVQ3_D4_1024',
        }[profile],
    )
    book = (nvq.codebook_for(spec).astype(np.int16) * 8).astype(np.int8)
    return nvq.NvqJscTensor(
        shape=(rows, width),
        axis=0,
        neuron_len=width,
        neuron_scale=rng.uniform(0.002, 0.02, rows).astype(np.float16).astype(np.float32),
        scale_lut=np.linspace(0.5, 1.5, 16, dtype=np.float16).astype(np.float32),
        bank_for_state=np.arange(16, dtype=np.uint8) & 1,
        state=rng.integers(0, 16, (rows, math.ceil(width / 24)), dtype=np.uint8),
        indices=rng.integers(
            0, spec.codebook_entries, (rows, math.ceil(width / spec.vector_size)), dtype=np.uint16
        ),
        signs=rng.integers(0, 128, (rows, math.ceil(width / 8)), dtype=np.uint8),
        codebooks=np.stack((book, book), axis=0),
        base_spec=spec,
    )


def _dense(tensor, modules):
    """Decode the reloaded payload using the original independent CPU formulas."""
    if isinstance(tensor, modules['mfq.formats.nint'].NintTensor):
        scale = tensor.neuron_scale[:, None] * tensor.sub_scale.astype(np.float32)
        minimum = tensor.neuron_min[:, None] * tensor.sub_min.astype(np.float32)
        dense = scale[:, :, None] * tensor.q.astype(np.float32) - minimum[:, :, None]
        return dense.reshape(tensor.q.shape[0], -1)[:, : tensor.neuron_len]
    if isinstance(tensor, modules['mfq.formats.nvq1_l'].Nvq1LTensor):
        return modules['mfq.quantize.nvq1_l_quant'].dequantize(tensor)
    if isinstance(tensor, modules['mfq.formats.nvq1_s'].Nvq1STensor):
        return modules['mfq.quantize.nvq1_s_quant'].dequantize(tensor)
    if isinstance(tensor, modules['mfq.formats.nvq'].NvqJscTensor):
        return modules['mfq.quantize.nvq_jsc'].dequantize_nvq_jsc(tensor)
    raise ValueError('Unexpected MFQ tensor profile')


def reference(case: Case, root: Path, source_root: Path) -> Path:
    """Recompute the CPU FP64 matrix reference from the saved MFQ payload and input."""
    if (
        case.data['op'] != 'mfq_cpu_quant_linear'
        or case.data['source'].get('generator') != 'tests/numerics/mfq_ref.py'
    ):
        raise ValueError('Unsupported MFQ reference case')
    modules, identity = _source(source_root)
    if identity != case.data['source']['mfq_source']:
        raise ValueError('MFQ source commit or module hashes differ from the case')
    case.asset('model')
    payload = case.asset('weight')
    io = modules['mfq.formats.io']
    _, model = io.load(case.root / case.data['assets']['model']['file'])
    if set(model) != {'linear.weight'}:
        raise ValueError('MFQ model must contain exactly the original linear weight')
    tensor = model['linear.weight']
    config = case.data['config']
    if config['experts'] > 1:
        mfe = modules['mfq.formats.mfe']
        if (
            not isinstance(tensor, mfe.MfeTensor)
            or tuple(tensor.shape) != (config['experts'], config['rows'], config['width'])
            or len(tensor.pools) != 1
            or list(tensor.pools[0].expert_ids) != config['expert_order']
            or config['expert_order'] != list(range(config['experts']))
        ):
            raise ValueError('MFQ model expert layout differs from the case')
        tensor = tensor.pools[0].tensor
    elif isinstance(tensor, modules['mfq.formats.mfe'].MfeTensor):
        raise ValueError('Unexpected MFQ expert container')
    dtype, model_payload = io.pack_tensor_payload(tensor)
    if dtype != config['payload_dtype'] or model_payload != payload:
        raise ValueError('MFQ model weight differs from the saved reference payload')
    dense = _dense(tensor, modules)
    expected_shape = (
        case.data['config']['experts'] * case.data['config']['rows'],
        case.data['config']['width'],
    )
    if dense.shape != expected_shape:
        raise ValueError('MFQ payload geometry differs from the case')
    value = case.tensor('input')
    expected = (value.astype(np.float64) @ dense.astype(np.float64).T).astype(np.float32)
    if _source(source_root)[1] != identity:
        raise ValueError('MFQ source changed during reference calculation')
    spec = save_tensor(root, 'output', expected)
    return write_result(
        case,
        root,
        {'output': spec},
        {
            'framework': 'NumPy',
            'numpy_version': np.__version__,
            'device': 'cpu',
            'arithmetic': 'float64 matmul, rounded to float32',
            'generator_sha256': _hash(__file__),
            'mfq_source': identity,
        },
    )


def generate(root: Path, source_root: Path, experts: int = 1) -> list[Path]:
    """Freeze the original 12 profiles and geometries using the specified MFQ checkout."""
    if type(experts) is not int or experts < 1:
        raise ValueError('Expert count must be a positive integer')
    modules, _ = _source(source_root)
    io = modules['mfq.formats.io']
    rng = np.random.default_rng(20261005)
    profiles = (
        'nint4',
        'nint5',
        'nint6',
        'nint8',
        'nvq1-s',
        'nvq1-l',
        'nvq2j',
        'nvq2j-l',
        'nvq2j-xl',
        'nvq3j',
        'nvq3j-512',
        'nvq3j-l',
    )
    original = Path(__file__).parents[2] / 'csrc/tests/make_cpu_quant_fixtures.py'
    paths = []
    for profile in profiles:
        for rows, width in ((640, 2560), (2560, 640)):
            tensor = _tensor(profile, rows * experts, width, rng, modules)
            dtype, payload = io.pack_tensor_payload(tensor)
            frozen = io.unpack_tensor_payload(dtype, payload)
            if experts == 1:
                stored = frozen
            else:
                mfe = modules['mfq.formats.mfe']
                stored = mfe.MfeTensor(
                    (experts, rows, width),
                    (mfe.MfePool(np.arange(experts, dtype=np.int32), frozen),),
                )
            value = rng.uniform(-0.15, 0.15, (3, width)).astype(np.float32)
            value[1] = 0
            value[2, width // 2] = 0.00003
            identity = _source(source_root)[1]
            name = f'{profile}-{rows}-{width}'
            writer = CaseWriter(
                root,
                name,
                'mfq_cpu_quant_linear',
                {
                    'profile': profile,
                    'rows': rows,
                    'width': width,
                    'experts': experts,
                    'expert_order': list(range(experts)),
                    'payload_dtype': dtype,
                    'model_dtype': dtype if experts == 1 else 'MFE',
                    'input_broadcast': 'same input rows for each flattened expert',
                    'native_checks': ['compressed_dispatch', 'row_slices']
                    if experts == 1
                    else ['expert_container_dispatch'],
                },
                {
                    'generator': 'tests/numerics/mfq_ref.py',
                    'generator_sha256': _hash(__file__),
                    'test': 'csrc/tests/make_cpu_quant_fixtures.py',
                    'test_sha256': _hash(original),
                    'seed': 20261005,
                    'mfq_source': identity,
                    'independence': 'MFQ Python/NumPy path does not call TyloInfer C++/CUDA decoder; shared format and implementation history',
                },
            )
            writer.asset('weight', payload, encoding=f'MFQ-{dtype}-payload')
            writer.input('input', value)
            model_path = writer.root / 'model.mfq'
            io.save(
                model_path,
                modules['mfq.formats.header'].FileHeader(version=2, model_arch='cpu-quant-oracle'),
                {'linear.weight': stored},
            )
            writer.asset('model', model_path, encoding='MFQv2')
            writer.output(
                'output',
                (3, rows * experts),
                'float32',
                {'kind': 'close', 'atol': 1e-5, 'rtol': 1e-3},
            )
            case = writer.finish()
            reference(case, case.root / 'reference', source_root)
            paths.append(case.root / 'case.json')
    return paths
