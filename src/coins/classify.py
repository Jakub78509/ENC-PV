"""Klasifikátor podle nejbližšího souseda se skóre pro každou třídu."""

import numpy as np


class NearestNeighborScores:
    """Pro každou třídu vzdálenost k nejbližšímu trénovacímu vzorku té třídy.

    Příznaky se standardizují (průměr 0, rozptyl 1 podle trénovací sady), aby průměr v mm
    a barva vážily srovnatelně. Skóre třídy = −vzdálenost, takže vyšší skóre = jistější;
    změnou prahu na tomto skóre se v kroku 06 počítá ROC a EER.
    """

    def __init__(self, k=1, weights=None):
        # weights: váha každého příznaku po standardizaci (např. (2, 1) = průměr v mm váží 2× víc
        # než sytost, protože velikost se mezi kamerami nemění, barva ano)
        self.k = k
        self.weights = weights

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        if self.weights is not None:
            self.std_ = self.std_ / np.asarray(self.weights, dtype=float)
        self.classes_ = np.unique(y)
        self.X_ = (X - self.mean_) / self.std_
        self.y_ = y
        return self

    def distances(self, X):
        """Matice (n_vzorků × n_tříd): průměr k nejmenších vzdáleností do každé třídy."""
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        out = np.empty((len(Z), len(self.classes_)))
        for j, c in enumerate(self.classes_):
            d = np.linalg.norm(Z[:, None, :] - self.X_[self.y_ == c][None, :, :], axis=2)
            out[:, j] = np.sort(d, axis=1)[:, : self.k].mean(axis=1)
        return out

    def decision_function(self, X):
        return -self.distances(X)

    def predict(self, X):
        return self.classes_[np.argmin(self.distances(X), axis=1)]
