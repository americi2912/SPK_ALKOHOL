import numpy as np

def normalize_weights(weights_raw: dict) -> dict:
    total = float(sum(weights_raw.values()))
    if total <= 0:
        raise ValueError("Total bobot harus > 0")
    return {k: float(v) / total for k, v in weights_raw.items()}

def topsis_rank(X: np.ndarray, w: np.ndarray, is_benefit: np.ndarray) -> dict:
    """
    TOPSIS:
    1) R = X / sqrt(sum(X^2) per kolom)
    2) Y = R * w
    3) A+ / A- berdasarkan benefit/cost
    4) D+ / D- (euclidean)
    5) V = D- / (D+ + D-)
    """
    if X.ndim != 2:
        raise ValueError("X harus 2D")

    m, n = X.shape
    if w.shape != (n,):
        raise ValueError("Dimensi bobot tidak sesuai jumlah kriteria")
    if is_benefit.shape != (n,):
        raise ValueError("Dimensi benefit/cost tidak sesuai jumlah kriteria")

    denom = np.sqrt((X ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    R = X / denom

    Y = R * w

    A_pos = np.empty(n, dtype=float)
    A_neg = np.empty(n, dtype=float)
    for j in range(n):
        if is_benefit[j]:
            A_pos[j] = Y[:, j].max()
            A_neg[j] = Y[:, j].min()
        else:
            A_pos[j] = Y[:, j].min()
            A_neg[j] = Y[:, j].max()

    D_plus = np.sqrt(((Y - A_pos) ** 2).sum(axis=1))
    D_minus = np.sqrt(((Y - A_neg) ** 2).sum(axis=1))

    denom_v = D_plus + D_minus
    denom_v = np.where(denom_v == 0, 1.0, denom_v)
    V = D_minus / denom_v

    return {
        "R": R, "Y": Y,
        "A_pos": A_pos, "A_neg": A_neg,
        "D_plus": D_plus, "D_minus": D_minus,
        "V": V
    }
