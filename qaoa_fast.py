"""
Швидкі (numba, паралельні) ядра точної симуляції QAOA для 18–21 кубіта.

Ті самі операції, що й qaoa_exact.qaoa_state / warm_start_qaoa.warm_state, але in-place і по блоках,
що влазять у кеш: на 21 кубіті один стан p=2 рахується за ~0.05–0.1 с замість ~2 с (numpy).
Конвенція бітів та сама: кубіт 0 — СТАРШИЙ біт індексу, тобто крок (stride) кубіта k = 2^(n-1-k).
Результат звіряється з qaoa_exact.qaoa_state у __main__ (і в tests/test_conventions.py).
"""
import numpy as np
import numba as nb

_BLOCK_BITS = 13          # блок 2^13 амплітуд (128 КБ) — молодші кубіти обробляються в кеші


@nb.njit(parallel=True, cache=True)
def _rx_inplace(psi, cs, ss, n):
    """RX на кожному кубіті: cs[e], ss[e] — cos і (−i·sin) кута для кубіта з кроком 2^e (e = n−1−k)."""
    N = psi.size
    b = min(n, _BLOCK_BITS)
    B = 1 << b
    for blk in nb.prange(N // B):
        base = blk * B
        for e in range(b):
            st = 1 << e
            c = cs[e]; s = ss[e]
            for i0 in range(0, B, 2 * st):
                for i in range(base + i0, base + i0 + st):
                    a0 = psi[i]; a1 = psi[i + st]
                    psi[i] = c * a0 + s * a1
                    psi[i + st] = s * a0 + c * a1
    for e in range(b, n):
        st = 1 << e
        c = cs[e]; s = ss[e]
        for t in nb.prange(N // 2):
            hi = t // st
            i = hi * 2 * st + (t - hi * st)
            a0 = psi[i]; a1 = psi[i + st]
            psi[i] = c * a0 + s * a1
            psi[i + st] = s * a0 + c * a1


@nb.njit(parallel=True, cache=True)
def _phase_inplace(psi, Ez, g):
    for i in nb.prange(psi.size):
        psi[i] *= np.exp(-1j * g * Ez[i])


@nb.njit(parallel=True, cache=True)
def expect(psi, Ez):
    """<Ez> = Σ |ψ|² · Ez."""
    acc = 0.0
    for i in nb.prange(psi.size):
        acc += (psi[i].real ** 2 + psi[i].imag ** 2) * Ez[i]
    return acc


def _mixer_coefs(beta, n, w=None):
    """w — ваги кубітів (індекс k, як у warm_start_qaoa.size_weights) або None."""
    w = np.ones(n) if w is None else np.asarray(w, float)
    ang = beta * w[::-1]                      # e = n−1−k
    return np.cos(ang).astype(np.complex128), (-1j * np.sin(ang)).astype(np.complex128)


def qaoa_state(Ez, params, n):
    """Холодний старт |+>^n → [фаза γ_l → RX(2β_l)] × p; params = [β_1..β_p, γ_1..γ_p] (як у qaoa_exact)."""
    p = len(params) // 2
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=np.complex128)
    for l in range(p):
        _phase_inplace(psi, Ez, float(params[p + l]))
        _rx_inplace(psi, *_mixer_coefs(float(params[l]), n), n)
    return psi


def warm_state(Ez, params, n, idx0, w=None):
    """Теплий старт |x0> → M(β0) → [P(γ_l) → M(β_l)] × p; params = [β0, γ_1..γ_p, β_1..β_p] (як у warm_start_qaoa)."""
    p = (len(params) - 1) // 2
    psi = np.zeros(2 ** n, dtype=np.complex128)
    psi[idx0] = 1.0
    _rx_inplace(psi, *_mixer_coefs(float(params[0]), n, w), n)
    for l in range(p):
        _phase_inplace(psi, Ez, float(params[1 + l]))
        _rx_inplace(psi, *_mixer_coefs(float(params[1 + p + l]), n, w), n)
    return psi


if __name__ == "__main__":
    import time
    from qaoa_exact import qaoa_state as qs_ref
    from warm_start_qaoa import warm_state as ws_ref
    rng = np.random.default_rng(0)
    for n in (5, 12, 15):
        Ez = rng.random(2 ** n)
        x = rng.uniform(0, 3, 4)
        assert np.allclose(qaoa_state(Ez, x, n), qs_ref(Ez, x, n))
        w = rng.uniform(0.2, 1, n)
        x = rng.uniform(0, 3, 5)
        assert np.allclose(warm_state(Ez, x, n, 7, w), ws_ref(Ez, x, n, 7, w))
    n = 21; Ez = rng.random(2 ** n); qaoa_state(Ez, [0.1, 0.2, 3, 4], n)
    t = time.time(); qaoa_state(Ez, [0.1, 0.2, 3, 4], n); print(f"n=21, p=2: {time.time() - t:.3f} с")
    print("ok: збігається з qaoa_exact / warm_start_qaoa")
