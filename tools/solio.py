"""Чтение двоичного файла решения, который пишут программы с ключом --out."""

import numpy as np


def read_solution(path):
    """Возвращает (x, y, w): узлы сетки и w[i, j] = w(x_i, y_j), включая границу."""
    with open(path, "rb") as fp:
        M, N = np.fromfile(fp, dtype=np.int32, count=2)
        A1, B1, A2, B2 = np.fromfile(fp, dtype=np.float64, count=4)
        w = np.fromfile(fp, dtype=np.float64, count=(M + 1) * (N + 1))
    x = A1 + (B1 - A1) / M * np.arange(M + 1)
    y = A2 + (B2 - A2) / N * np.arange(N + 1)
    return x, y, w.reshape(M + 1, N + 1)


def in_D(x, y):
    """Область варианта 12: x^2 - 4y^2 > 1, 1 < x < 3."""
    return (x * x - 4 * y * y > 1) & (x > 1) & (x < 3)


def bilinear(x, y, w, px, py):
    """Билинейная интерполяция сеточной функции в точку (px, py)."""
    i = int(np.clip(np.searchsorted(x, px) - 1, 0, len(x) - 2))
    j = int(np.clip(np.searchsorted(y, py) - 1, 0, len(y) - 2))
    tx = (px - x[i]) / (x[i + 1] - x[i])
    ty = (py - y[j]) / (y[j + 1] - y[j])
    return ((1 - tx) * (1 - ty) * w[i, j] + tx * (1 - ty) * w[i + 1, j]
            + (1 - tx) * ty * w[i, j + 1] + tx * ty * w[i + 1, j + 1])
