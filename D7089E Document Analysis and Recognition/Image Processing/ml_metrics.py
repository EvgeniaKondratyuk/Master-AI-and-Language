# ------------------------------------------------------------
# D7089 Lecture 6 – Decision trees, SVMs, data and evaluation metrics
# Runnable examples for every topic in the lecture (tested)
# ------------------------------------------------------------
import numpy as np
from sklearn.datasets import make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, precision_recall_curve,
                             average_precision_score)

rng = np.random.default_rng(0)

# ============================================================
# 1. Decision tree: questions on features, rectangular boundaries
# ============================================================
X, y = make_classification(n_samples=600, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=2, class_sep=0.8, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

full_tree = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X_tr, y_tr)   # grows until pure
pruned_tree = DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.01,                   # cost-complexity pruning
                                     random_state=0).fit(X_tr, y_tr)
for name, t in (("full tree", full_tree), ("pruned tree", pruned_tree)):
    print(f"{name:12s} depth={t.get_depth():2d} leaves={t.get_n_leaves():3d} "
          f"train acc={t.score(X_tr, y_tr):.2f}  test acc={t.score(X_te, y_te):.2f}")
print(export_text(pruned_tree, feature_names=["width", "height"]))          # the "20 questions"


def entropy(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


# Information gain of a question that splits 8 equally likely animals 4/4 vs 1/7
H0 = entropy([1 / 8] * 8)                                          # 3 bits
gain_half = H0 - (0.5 * entropy([1 / 4] * 4) + 0.5 * entropy([1 / 4] * 4))
gain_one = H0 - (1 / 8 * 0 + 7 / 8 * entropy([1 / 7] * 7))
print(f"start entropy {H0:.2f} bits | gain 4/4 split {gain_half:.2f} | gain 1/7 split ('is it a dog?') {gain_one:.2f}")

# ============================================================
# 2. SVM: maximum margin, soft margin, multiclass, nonlinear mapping
# ============================================================
Xc, yc = make_circles(n_samples=400, factor=0.4, noise=0.05, random_state=0)   # concentric circles
lin = LinearSVC().fit(Xc, yc)
print("linear SVM on circles (x1, x2):", round(lin.score(Xc, yc), 2))

# (a) coordinate transform to polar: r = sqrt(x1^2 + x2^2), theta = atan2(x2, x1)
polar = np.column_stack([np.hypot(Xc[:, 0], Xc[:, 1]), np.arctan2(Xc[:, 1], Xc[:, 0])])
print("linear SVM on (r, theta):     ", round(LinearSVC().fit(polar, yc).score(polar, yc), 2))

# (b) feature mapping: add x1^2, x2^2, x1*x2 columns ("more columns in the spreadsheet")
mapped = np.column_stack([Xc, Xc[:, 0] ** 2, Xc[:, 1] ** 2, Xc[:, 0] * Xc[:, 1]])
print("linear SVM on mapped features:", round(LinearSVC().fit(mapped, yc).score(mapped, yc), 2))

# (c) the kernel trick does the mapping implicitly
rbf = SVC(kernel="rbf", C=1.0).fit(Xc, yc)
print("RBF-kernel SVM:", round(rbf.score(Xc, yc), 2), "| support vectors:", rbf.n_support_.sum(), "of", len(Xc))

# Soft margin: C trades margin width against training errors
for C in (0.01, 1, 100):
    m = SVC(kernel="linear", C=C).fit(X_tr, y_tr)
    print(f"linear SVM C={C:<6} support vectors={m.n_support_.sum():3d} test acc={m.score(X_te, y_te):.2f}")

# Multiclass: one-vs-rest (k classifiers) vs one-vs-one (k(k-1)/2 classifiers)
X3, y3 = make_classification(n_samples=600, n_features=4, n_informative=3, n_redundant=0, n_classes=3,
                             n_clusters_per_class=1, random_state=1)
ovr = OneVsRestClassifier(SVC(kernel="linear")).fit(X3, y3)
ovo = OneVsOneClassifier(SVC(kernel="linear")).fit(X3, y3)
print("OvR classifiers:", len(ovr.estimators_), "| OvO classifiers:", len(ovo.estimators_))
print("26 letters -> OvR", 26, "classifiers, OvO", 26 * 25 // 2)

# ============================================================
# 3. Feature normalization (why units matter for kNN)
# ============================================================
width_px = rng.normal(10, 2, 200)
height_px = rng.normal(14, 2, 200)
labels = (width_px + height_px > 24).astype(int)
raw = np.column_stack([width_px * 1e6 / 300 * 25.4e-3,      # width in micrometres (300 dpi)
                       height_px / 300 * 0.0254])            # height in metres
for name, feats in (("raw (um vs m)", raw),
                    ("min-max [0,1]", MinMaxScaler().fit_transform(raw)),
                    ("z-score", StandardScaler().fit_transform(raw))):
    a, b, ya, yb = train_test_split(feats, labels, test_size=0.3, random_state=0)
    print(f"kNN with {name:14s} test acc = {KNeighborsClassifier(3).fit(a, ya).score(b, yb):.2f}")
# NB: in a real project fit the scaler on the TRAINING split only, then apply it to val/test.

# ============================================================
# 4. Train / validation / test split: 80 / 10 / 10
# ============================================================
idx = np.arange(1000)
train_idx, rest = train_test_split(idx, test_size=0.2, random_state=0)
val_idx, test_idx = train_test_split(rest, test_size=0.5, random_state=0)
print("split sizes:", len(train_idx), len(val_idx), len(test_idx))

# ============================================================
# 5. Confusion matrix, accuracy, precision, recall, F1
# ============================================================
y_true = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
y_pred = ["A", "A", "B",  "B", "B", "C",  "C", "C", "A"]
cm = confusion_matrix(y_true, y_pred, labels=["A", "B", "C"])
print("confusion matrix (rows = true, cols = predicted)\n", cm)
print("errors (off-diagonal):", cm.sum() - np.trace(cm),
      "| accuracy:", round(accuracy_score(y_true, y_pred), 3))

# Binary detection example: "is this region handwritten?"
y_t = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
y_p = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])      # TP=3, FN=1, FP=2, TN=4
P, R, F = precision_score(y_t, y_p), recall_score(y_t, y_p), f1_score(y_t, y_p)
print(f"precision={P:.2f} recall={R:.2f} F1={F:.3f} (check 2PR/(P+R) = {2 * P * R / (P + R):.3f})")

# Precision-recall curve from confidence scores, and average precision (area under it)
scores = np.array([0.95, 0.9, 0.8, 0.35, 0.7, 0.6, 0.3, 0.2, 0.15, 0.1])
prec, rec, thr = precision_recall_curve(y_t, scores)
for p, r, t in zip(prec, rec, np.append(thr, np.nan)):
    print(f"  threshold {t:4.2f}: precision {p:.2f} recall {r:.2f}")
print("average precision (AP):", round(average_precision_score(y_t, scores), 3))

# ============================================================
# 6. Character error rate (CER) and word error rate (WER)
# ============================================================
def edit_ops(ref, hyp):
    """Minimum (substitutions, deletions, insertions) turning ref into hyp (Levenshtein)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1,                               # deletion
                          D[i, j - 1] + 1,                               # insertion
                          D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))  # substitution / match
    # backtrack to count each error type
    i, j, S, Dl, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]; i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            Dl += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, Dl, I


def error_rate(ref, hyp):
    S, Dl, I = edit_ops(ref, hyp)
    return (S + Dl + I) / len(ref), (S, Dl, I)


def cer(ref, hyp):
    return error_rate(list(ref), list(hyp))


def wer(ref, hyp):
    return error_rate(ref.split(), hyp.split())


print("CER machine learning -> machin leaming:", cer("machine learning", "machin leaming"))
print("WER same pair:", wer("machine learning", "machin leaming"))
print("WER fox:", wer("the quick brown fox jumps over the lazy dog",
                      "the brown fox jumps again over the dog"))
print("CER can exceed 100%:", cer("cat", "the cat sat on the mat"))
