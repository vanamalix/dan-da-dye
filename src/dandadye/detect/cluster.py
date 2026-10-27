"""Chapter-wide character clustering on unit-norm crop embeddings."""

from __future__ import annotations

from collections import Counter

import numpy as np


def cluster_embeddings(embs, threshold: float = 0.55) -> list[int]:
    """Average-linkage agglomerative clustering; merge while mean cosine sim > threshold."""
    x = np.asarray(embs, dtype=np.float64)
    n = len(x)
    if n == 0:
        return []
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    sim = x @ x.T
    clusters = [[i] for i in range(n)]
    # cluster-level similarity sums; merge greedily
    S = sim.copy()
    sizes = np.ones(n)
    active = np.ones(n, dtype=bool)
    np.fill_diagonal(S, -np.inf)
    while active.sum() > 1:
        avg = S / np.outer(sizes, sizes)
        avg[~active, :] = -np.inf
        avg[:, ~active] = -np.inf
        i, j = np.unravel_index(np.argmax(avg), avg.shape)
        if avg[i, j] <= threshold:
            break
        S[i, :] += S[j, :]
        S[:, i] += S[:, j]
        S[i, i] = -np.inf
        sizes[i] += sizes[j]
        active[j] = False
        clusters[i] += clusters[j]
        clusters[j] = []
    labels = [0] * n
    for lab, members in enumerate(c for c in clusters if c):
        for m in members:
            labels[m] = lab
    return labels


def cluster_votes(labels: list[int], names: list[str], min_share: float = 0.6, min_votes: int = 2) -> dict:
    """Cluster -> name when >= min_votes members are named and min_share of those agree."""
    by: dict[int, list[str]] = {}
    for lab, name in zip(labels, names):
        by.setdefault(lab, []).append(name)
    out = {}
    for lab, ns in by.items():
        named = [n for n in ns if n and n != "Other"]
        if len(named) >= min_votes:
            top, cnt = Counter(named).most_common(1)[0]
            if cnt / len(named) >= min_share:
                out[lab] = top
    return out
