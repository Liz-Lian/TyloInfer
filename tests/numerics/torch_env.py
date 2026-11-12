"""Own Torch reference execution settings without leaking them between cases."""

from contextlib import contextmanager

import numpy as np
import torch


def device(value, kind='cuda'):
    """Resolve the concrete device required by the original reference case."""
    result = torch.device(value)
    if result.type != kind:
        raise ValueError(f'This reference requires a {kind} device')
    if kind == 'cuda' and result.index is None:
        result = torch.device('cuda', torch.cuda.current_device())
    return result


def settings():
    """Read the Torch matmul settings that affect reference arithmetic."""
    return {
        'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }


def _apply(config):
    """Apply recorded settings and reject configurations Torch cannot reproduce."""
    flags = (
        'allow_tf32',
        'allow_fp16_reduced_precision_reduction',
        'allow_bf16_reduced_precision_reduction',
    )
    if (
        set(config) != {*flags, 'float32_matmul_precision'}
        or any(type(config[name]) is not bool for name in flags)
        or config['float32_matmul_precision'] not in {'highest', 'high', 'medium'}
    ):
        raise ValueError('Invalid Torch reference compute configuration')
    torch.set_float32_matmul_precision(config['float32_matmul_precision'])
    if torch.backends.cuda.matmul.allow_tf32 != config['allow_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = config['allow_tf32']
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = config[
        'allow_fp16_reduced_precision_reduction'
    ]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config[
        'allow_bf16_reduced_precision_reduction'
    ]
    if settings() != config:
        raise ValueError('Recorded Torch matmul settings cannot be reproduced')


@contextmanager
def scope(config):
    """Use a case's recorded settings and restore the caller's settings on exit."""
    previous = settings()
    try:
        _apply(config)
        yield
    finally:
        _apply(previous)


def execution(target, accumulation):
    """Describe the framework, concrete device and current arithmetic settings."""
    result = {
        'framework': 'PyTorch',
        'torch_version': str(torch.__version__),
        'numpy_version': np.__version__,
        'device': str(target),
        'cuda_version': torch.version.cuda,
        'accumulation': accumulation,
        'compute': settings(),
    }
    if target.type == 'cuda':
        result.update(
            device_name=torch.cuda.get_device_name(target),
            device_capability=list(torch.cuda.get_device_capability(target)),
        )
    return result
