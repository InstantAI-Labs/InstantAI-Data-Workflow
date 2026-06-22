from __future__ import annotations

def validate_split_ratios(val_ratio: float, test_ratio: float = 0.0) -> None:
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError(f'split ratios must be non-negative: val={val_ratio} test={test_ratio}')
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f'val_ratio + test_ratio must be < 1.0 (train needs a positive share); '
            f'got val={val_ratio} test={test_ratio}'
        )

def split_unit_fraction(key: str, *, seed: int) -> float:
    from indw.util.stable_hash import stable_digest_int
    bits = stable_digest_int({'seed': int(seed), 'key': str(key)}, bits=53)
    return bits / float(1 << 53)

def assign_split_for_key(
    key: str,
    *,
    val_ratio: float,
    test_ratio: float = 0.0,
    seed: int,
) -> str:
    return assign_split(
        split_unit_fraction(key, seed=seed),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

def assign_split(u: float, *, val_ratio: float, test_ratio: float = 0.0) -> str:

    validate_split_ratios(val_ratio, test_ratio)
    if u < test_ratio:
        return 'test'
    if u < test_ratio + val_ratio:
        return 'val'
    return 'train'
