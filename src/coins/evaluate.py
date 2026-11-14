"""ROC, EER a AUC pro každou hodnotu mince (jedna třída proti ostatním)."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def eer(fpr, tpr):
    """Equal Error Rate: bod, kde FAR (= FPR) = FRR (= 1 − TPR), lineární interpolace mezi body ROC."""
    frr = 1 - tpr
    diff = fpr - frr
    i = np.where(diff >= 0)[0][0]  # první práh, kde FAR už dohnala FRR
    if i == 0 or diff[i] == 0:
        return float(fpr[i])
    t = diff[i - 1] / (diff[i - 1] - diff[i])  # průsečík úsečky mezi body i−1 a i
    return float(fpr[i - 1] + t * (fpr[i] - fpr[i - 1]))


def roc_per_class(y_true, scores, classes):
    """Pro každou třídu (fpr, tpr) a tabulku s EER, AUC a počtem pozitivních/negativních vzorků.

    scores: matice (n_vzorků × n_tříd), sloupce ve stejném pořadí jako classes; vyšší = jistější.
    """
    y_true = np.asarray(y_true)
    curves, rows = {}, []
    for j, c in enumerate(classes):
        positive = y_true == c
        fpr, tpr, _ = roc_curve(positive, scores[:, j])
        curves[c] = (fpr, tpr)
        rows.append({"hodnota": c, "EER": eer(fpr, tpr), "AUC": roc_auc_score(positive, scores[:, j]),
                     "pozitivních": int(positive.sum()), "negativních": int((~positive).sum())})
    return curves, pd.DataFrame(rows).set_index("hodnota")
