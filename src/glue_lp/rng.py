def mulberry32(seed: int):
    t = seed & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal t
        t = (t + 0x6D2B79F5) & 0xFFFFFFFF
        r = (t ^ (t >> 15)) * (1 | t)
        r &= 0xFFFFFFFF
        r ^= r + ((r ^ (r >> 7)) * (61 | r))
        r &= 0xFFFFFFFF
        return ((r ^ (r >> 14)) & 0xFFFFFFFF) / 4294967296

    return rnd


def shuffle(arr: list, rnd) -> None:
    for i in range(len(arr) - 1, 0, -1):
        j = int(rnd() * (i + 1))
        arr[i], arr[j] = arr[j], arr[i]
