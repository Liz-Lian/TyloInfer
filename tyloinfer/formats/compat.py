from __future__ import annotations

LEGACY_DTYPES = frozenset({'NINT', 'NINT8-0', 'NVQ', 'NPQ', 'NEPQ', 'MFE', 'MFED',
    'MXFP4-SQ', 'MXFP8-SQ', 'FP8-128SQ'})
_ALIASES = {
    **{f'NINT{q}': 'NINT' for q in range(1, 9)}, 'NINTv2': 'NINT',
    'NINTM': 'MFE', 'NINTMD': 'MFED',
    **{name: 'NVQ' for name in ('NIQ2', 'NIQ2J', 'NIQ3', 'NVQ1-L', 'NVQ1-S',
        'NVQ2', 'NVQ2J', 'NVQ2J-L', 'NVQ2J-XL', 'NVQ3', 'NVQ3J', 'NVQ3J-512', 'NVQ3J-L')},
    **{name: 'NPQ' for name in ('NPQ0-L', 'NPQ0-S')},
    **{name: 'NEPQ' for name in ('NEPQ0-L', 'NEPQ0-S', 'NEPQ1-L', 'NEPQ1-S', 'NEPQ0-A', 'NEPQ1-A')},
    'MXFP4-SQ2': 'MXFP4-SQ', 'MXFP4-SQ3': 'MXFP4-SQ',
}


def canonical_dtype(dtype: str) -> str:
    return _ALIASES.get(dtype, dtype)
