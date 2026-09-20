from __future__ import annotations

def leakage_exists(mp_keys, pos_keys):
    return bool(set(mp_keys) & set(pos_keys))

def assert_no_leakage(mp_keys, pos_keys):
    if leakage_exists(mp_keys, pos_keys):
        raise AssertionError("invariante violada: aresta-alvo em E_mp")
