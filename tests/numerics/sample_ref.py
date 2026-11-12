"""Fixed sampling and embedding references from the existing CUDA tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .case import Case, CaseWriter, save_tensor, write_result
from .torch_env import device as resolve_device
from .torch_env import execution, scope, settings


def _array(value):
    """Expand BF16 exactly when converting a tensor to NumPy."""
    value = value.detach().cpu()
    return (value.float() if value.dtype == torch.bfloat16 else value).numpy()


def _input(writer, name, value):
    """Persist an input without losing its original logical dtype."""
    dtype = 'bfloat16' if value.dtype == torch.bfloat16 else None
    writer.input(name, _array(value), dtype=dtype)


def _tensor(case, name, device):
    """Read a fixed input and restore its logical Torch dtype."""
    dtype = case.data['inputs'][name]['dtype']
    return torch.tensor(case.tensor(name), device=device, dtype=getattr(torch, dtype))


def _writer(root, name, op, config, test):
    """Create a case with source and generator content identities."""
    source = Path(__file__).parents[1] / 'test_kernels_cuda.py'
    return CaseWriter(
        root,
        name,
        op,
        {**config, 'compute': settings()},
        {
            'file': 'tests/test_kernels_cuda.py',
            'test': test,
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'generator': 'tests/numerics/sample_ref.py',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )


def _fixed(case, device):
    """Return original hand-specified results only for their exact source inputs."""
    op = case.data['op']
    if op == 'sample_greedy':
        expected_input = torch.tensor(
            [[1.0, 5.0, 5.0, -1.0], [0.0, -2.0, 3.0, 1.0]], device=device, dtype=torch.float32
        )
        if not torch.equal(_tensor(case, 'logits', device).float(), expected_input):
            raise ValueError('Greedy fixed baseline requires the original logits')
        return {'output': torch.tensor([1, 2], device=device, dtype=torch.int64)}
    if op == 'sample_tie':
        if (
            not torch.equal(
                _tensor(case, 'logits', device),
                torch.ones((1, 8), device=device, dtype=torch.float32),
            )
            or not torch.equal(
                _tensor(case, 'random', device),
                torch.tensor([0.9], device=device, dtype=torch.float32),
            )
            or any(
                case.data['config'].get(k) != v
                for k, v in [('temperature', 1.0), ('top_k', 3), ('top_p', 1.0)]
            )
        ):
            raise ValueError('Tie-break fixed baseline requires the original inputs')
        return {'output': torch.tensor([2], device=device, dtype=torch.int64)}
    counts = _tensor(case, 'counts', device)
    tokens = _tensor(case, 'tokens', device)
    logits = _tensor(case, 'logits', device)
    if (
        not torch.equal(counts, torch.zeros(5, device=device, dtype=torch.int32))
        or not torch.equal(tokens, torch.tensor([1, 1, 3], device=device, dtype=torch.int64))
        or not torch.equal(
            logits, torch.tensor([[2.0, -2.0, 1.0, 3.0, 4.0]], device=device, dtype=torch.float32)
        )
        or any(
            case.data['config'].get(k) != v
            for k, v in [('presence', 0.5), ('frequency', 0.25), ('repetition', 2.0)]
        )
    ):
        raise ValueError('Penalty fixed baseline requires the original inputs')
    return {
        'counts': torch.tensor([0, 2, 0, 1, 0], device=device, dtype=torch.int32),
        'output': torch.tensor([[2.0, -5.0, 1.0, 0.75, 4.0]], device=device, dtype=torch.float32),
    }


def _calculate(case, device):
    """Apply the original Torch references or validated fixed expectations."""
    op, config = case.data['op'], case.data['config']
    if op in ('sample_greedy', 'sample_tie', 'sample_penalties'):
        return _fixed(case, device)
    if op == 'embedding':
        return {'output': _tensor(case, 'weight', device)[_tensor(case, 'ids', device)]}
    logits, rnd = _tensor(case, 'logits', device), _tensor(case, 'random', device)
    if op == 'sample_softmax':
        probs = torch.softmax(logits / config['temperature'], dim=-1)
        output = torch.searchsorted(torch.cumsum(probs, dim=-1), rnd[:, None]).squeeze(1)
        return {'output': output.to(torch.int64)}
    if op == 'sample_topk_topp':
        if config['temperature'] != 1.0:
            raise ValueError('The original top-k/top-p reference requires temperature 1')
        vals, idx = torch.topk(logits, k=config['top_k'], dim=-1)
        probs = torch.softmax(vals, dim=-1)
        refs = []
        for batch in range(logits.size(0)):
            cumulative = torch.cumsum(probs[batch], dim=0)
            keep = int((cumulative >= config['top_p']).nonzero()[0].item()) + 1
            kept = probs[batch, :keep]
            kept = kept / kept.sum()
            selected = int(torch.searchsorted(torch.cumsum(kept, dim=0), rnd[batch]).item())
            refs.append(int(idx[batch, selected].item()))
        return {'output': torch.tensor(refs, device=device, dtype=torch.int64)}
    raise ValueError(f'Unsupported sampling reference operation: {op!r}')


def reference(case: Case, root: Path, device: str = 'cuda') -> Path:
    """Recalculate one source case from persisted inputs on CUDA."""
    device = resolve_device(device)
    if case.data['source'].get('generator') != 'tests/numerics/sample_ref.py':
        raise ValueError('Unsupported sampling reference source')
    with scope(case.data['config']['compute']):
        outputs = _calculate(case, device)
        target = Path(root)
        specs = {name: save_tensor(target, name, _array(value)) for name, value in outputs.items()}
        metadata = execution(device, 'source Torch dtype and reduction semantics')
        metadata['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return write_result(case, target, specs, metadata)


def _publish(writer, device):
    """Publish fixed inputs and calculate their reference result."""
    case = writer.finish()
    reference(case, case.root / 'reference', device)
    return case.root / 'case.json'


def _samples(root, device):
    """Create the original sampling and penalty cases."""
    results = []
    logits = torch.tensor(
        [[1.0, 5.0, 5.0, -1.0], [0.0, -2.0, 3.0, 1.0]], device=device, dtype=torch.float32
    )
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        name = str(dtype).removeprefix('torch.')
        writer = _writer(
            root, f'greedy_{name}', 'sample_greedy', {}, 'test_sample_greedy_cuda_f32_f16_bf16'
        )
        _input(writer, 'logits', logits.to(dtype))
        writer.output('output', (2,), 'int64', {'kind': 'exact'})
        results.append(_publish(writer, device))
    writer = _writer(
        root,
        'sample_softmax',
        'sample_softmax',
        {'temperature': 0.7},
        'test_sample_softmax_cuda_matches_reference',
    )
    _input(
        writer,
        'logits',
        torch.tensor(
            [[0.1, 0.4, -0.2, 1.0], [2.0, -1.0, 0.0, 0.5]], device=device, dtype=torch.float32
        ),
    )
    _input(writer, 'random', torch.tensor([0.10, 0.95], device=device, dtype=torch.float32))
    writer.output('output', (2,), 'int64', {'kind': 'exact'})
    results.append(_publish(writer, device))
    writer = _writer(
        root,
        'sample_topk_topp',
        'sample_topk_topp',
        {'temperature': 1.0, 'top_k': 3, 'top_p': 0.75},
        'test_sample_top_k_top_p_cuda_matches_reference',
    )
    _input(
        writer,
        'logits',
        torch.tensor(
            [[0.1, 2.0, 1.2, -0.5, 0.8], [1.0, 0.9, 0.1, 2.5, -1.0]],
            device=device,
            dtype=torch.float32,
        ),
    )
    _input(writer, 'random', torch.tensor([0.4, 0.8], device=device, dtype=torch.float32))
    writer.output('output', (2,), 'int64', {'kind': 'exact'})
    results.append(_publish(writer, device))
    writer = _writer(
        root,
        'sample_tie',
        'sample_tie',
        {'temperature': 1.0, 'top_k': 3, 'top_p': 1.0},
        'test_sample_top_k_cuda_tie_breaks_by_lowest_token_id',
    )
    _input(writer, 'logits', torch.ones((1, 8), device=device, dtype=torch.float32))
    _input(writer, 'random', torch.tensor([0.9], device=device, dtype=torch.float32))
    writer.output('output', (1,), 'int64', {'kind': 'exact'})
    results.append(_publish(writer, device))
    writer = _writer(
        root,
        'sample_penalties',
        'sample_penalties',
        {
            'presence': 0.5,
            'frequency': 0.25,
            'repetition': 2.0,
            'native_checks': ['in_place_pointer'],
        },
        'test_sample_penalties_cuda_counts_and_updates_in_place',
    )
    _input(writer, 'counts', torch.zeros(5, device=device, dtype=torch.int32))
    _input(writer, 'tokens', torch.tensor([1, 1, 3], device=device, dtype=torch.int64))
    _input(
        writer,
        'logits',
        torch.tensor([[2.0, -2.0, 1.0, 3.0, 4.0]], device=device, dtype=torch.float32),
    )
    writer.output('counts', (5,), 'int32', {'kind': 'exact'})
    writer.output('output', (1, 5), 'float32', {'kind': 'exact'})
    results.append(_publish(writer, device))
    return results


def _embeddings(root, device):
    """Create the original seeded float16 and float32 embedding cases."""
    results = []
    cases = [
        ('f16', 26, (17, 13), torch.float16, [[0, 3, 16], [8, 2, 5]]),
        ('f32', 27, (19, 11), torch.float32, [18, 1, 7, 0]),
    ]
    for suffix, seed, shape, dtype, ids in cases:
        torch.manual_seed(seed)
        weight = torch.randn(*shape, device=device, dtype=dtype)
        indices = torch.tensor(ids, device=device, dtype=torch.int64)
        writer = _writer(
            root,
            f'embedding_{suffix}',
            'embedding',
            {'seed': seed},
            f'test_embedding_lookup_cuda_{suffix}',
        )
        _input(writer, 'weight', weight)
        _input(writer, 'ids', indices)
        writer.output(
            'output',
            (*indices.shape, shape[1]),
            str(dtype).removeprefix('torch.'),
            {'kind': 'exact'},
        )
        results.append(_publish(writer, device))
    return results


def generate(root: Path, suite: str = 'sample', device: str = 'cuda') -> list[Path]:
    """Generate the original sampling or embedding cases from fixed inputs."""
    device = resolve_device(device)
    if suite == 'sample':
        return _samples(root, device)
    if suite == 'embedding':
        return _embeddings(root, device)
    raise ValueError(f'Unknown sampling suite: {suite!r}')
