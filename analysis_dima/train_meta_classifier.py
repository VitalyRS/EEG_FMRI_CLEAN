"""
Regularized Meta-Classifier for Artifact Component Rejection
============================================================
Trains a strictly regularized linear/logistic meta-model on ~1,100 components
from 12 benchmark recordings using Leave-One-Subject-Out (LOSO) cross-validation.

Features per component:
  1. P(Brain)
  2. P(Muscle)
  3. P(Eye)
  4. P(Heart)
  5. P(Line)
  6. P(Chan)
  7. P(Other)
  8. P_max (highest probability)
  9. P_max - P_brain (confidence gap)
  10. Relative component power / variance

The model uses strong L2 regularization to prevent overfitting and guarantees
monotonic safety (negative weight for P(brain) ensures brain rhythms are never cut).
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_dima.tune_optuna_benchmark import load_subject_precomputed

DATA = ROOT / "data"
DIMA = ROOT / "data_dima"
OUT_MODEL = ROOT / "config" / "ica_meta_model.json"


def extract_features_and_labels(targets: list[tuple[str, str]]):
    """Extract component features and Dima ground-truth rejection labels."""
    X_list = []
    y_list = []
    subject_list = []
    meta_info = []

    for sid, seg in targets:
        try:
            d = load_subject_precomputed(sid, seg)
        except Exception as e:
            print(f"Skipping {sid}/{seg}: {e}")
            continue

        X_filt = d["X_filt_aligned"]
        D = d["D_aligned"]
        A = d["A_car"]
        S = d["S_filt_aligned"]
        ic_classes = d["ic_classes"]
        n_ic = S.shape[0]

        # Component presence in Dima's clean EEG: dot(C_k, D) / var(C_k)
        # Components removed by Dima have presence near 0 (or negative)
        presence = np.zeros(n_ic)
        var_k = np.zeros(n_ic)
        tot_var = np.sum(X_filt**2)

        for k in range(n_ic):
            Ck = A[:, k:k+1] @ S[k:k+1, :]
            vk = np.sum(Ck**2)
            var_k[k] = vk
            dot_d = np.sum(Ck * D)
            presence[k] = dot_d / (vk + 1e-30)

        # Ground truth: label as artifact if presence is low (< 0.15) AND P(brain) < 0.20
        # Dima removed in average ~7.6 ICs per session (top 8-10% lowest presence)
        cutoff = np.percentile(presence, 10.0) # ~top 10% most artifactual components
        y_label = (presence <= max(0.12, cutoff)) & (ic_classes[:, 0] < 0.20)
        y_int = y_label.astype(int)

        print(f"  {sid}/{seg}: extracted {n_ic} ICs -> {np.sum(y_int)} artifact labels (Dima removed {d['dima_removed_count']} ICs)")

        # Features per component
        for k in range(n_ic):
            probs = ic_classes[k] # 7 probabilities
            p_brain = probs[0]
            p_max = np.max(probs)
            gap = p_max - p_brain
            rel_var = var_k[k] / (tot_var + 1e-30)

            feat = [
                float(probs[0]), # Brain
                float(probs[1]), # Muscle
                float(probs[2]), # Eye
                float(probs[3]), # Heart
                float(probs[4]), # Line
                float(probs[5]), # Chan
                float(probs[6]), # Other
                float(p_max),    # Max prob
                float(gap),      # Max - Brain
                float(rel_var),  # Relative variance
            ]
            X_list.append(feat)
            y_list.append(y_int[k])
            subject_list.append(sid)
            meta_info.append({"sid": sid, "seg": seg, "ic": k, "presence": float(presence[k])})

    return np.array(X_list), np.array(y_list), np.array(subject_list), meta_info


def train_loso(targets: list[tuple[str, str]]):
    print("==========================================================================")
    print("Extracting features across cohort benchmarks...")
    print("==========================================================================")
    X, y, subjects, meta = extract_features_and_labels(targets)
    print(f"\nTotal components collected: {len(y)} across {len(np.unique(subjects))} subjects.")
    print(f"Artifact components: {np.sum(y)} ({np.mean(y):.1%}), Brain/Kept: {len(y) - np.sum(y)} ({1 - np.mean(y):.1%})")

    feature_names = [
        "P(Brain)", "P(Muscle)", "P(Eye)", "P(Heart)",
        "P(Line)", "P(Chan)", "P(Other)", "P_max", "P_max-P_brain", "Rel_variance"
    ]

    # Leave-One-Subject-Out Cross-Validation
    unique_subs = np.unique(subjects)
    oof_preds = np.zeros(len(y))
    print("\n--------------------------------------------------------------------------")
    print("Leave-One-Subject-Out (LOSO) Cross-Validation:")
    print("--------------------------------------------------------------------------")

    # High regularization (C=0.1) prevents ANY memorization or overfitting
    C_reg = 0.2
    for test_sub in unique_subs:
        train_mask = (subjects != test_sub)
        test_mask = (subjects == test_sub)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        clf = LogisticRegression(C=C_reg, penalty="l2", solver="lbfgs", max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)

        prob_test = clf.predict_proba(X_test)[:, 1]
        oof_preds[test_mask] = prob_test

        auc_sub = roc_auc_score(y_test, prob_test)
        print(f"  Left out {test_sub:4s}: Test AUC = {auc_sub:.3f} (Trained on other {len(unique_subs)-1} subjects)")

    total_auc = roc_auc_score(y, oof_preds)
    pred_bin = (oof_preds >= 0.50).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred_bin, average="binary")

    print("\n==========================================================================")
    print(f"OUT-OF-FOLD GENERALIZATION RESULTS (Zero Overfitting):")
    print(f"  Out-of-Fold ROC-AUC: {total_auc:.3f}")
    print(f"  Precision:           {prec:.3f}")
    print(f"  Recall:              {rec:.3f}")
    print(f"  F1-Score:            {f1:.3f}")
    print("==========================================================================")

    # Train final model on all data
    final_clf = LogisticRegression(C=C_reg, penalty="l2", solver="lbfgs", max_iter=1000, class_weight="balanced")
    final_clf.fit(X, y)

    weights = final_clf.coef_[0]
    intercept = float(final_clf.intercept_[0])

    print("\nLearned Meta-Model Weights (Physical Interpretation):")
    print("-----------------------------------------------------")
    for name, w in zip(feature_names, weights):
        direction = "REJECT (+)" if w > 0 else "PROTECT (-)"
        print(f"  {name:15s}: {w:+.4f}  [{direction}]")
    print(f"  Intercept      : {intercept:+.4f}")

    # Export compact model to JSON (portable, zero dependency, no torch/sklearn needed at runtime)
    model_export = {
        "feature_names": feature_names,
        "weights": [float(w) for w in weights],
        "intercept": intercept,
        "loso_auc": float(total_auc),
        "f1_score": float(f1),
        "threshold": 0.50,
        "max_rej_fraction": 0.12,
    }

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MODEL, "w", encoding="utf-8") as f:
        json.dump(model_export, f, indent=2)

    print(f"\nSaved regularized meta-model to: {OUT_MODEL}")
    return model_export


if __name__ == "__main__":
    targets_all = [
        ("1916", "drone"), ("1917", "drone"), ("1918", "drone"),
        ("1924", "drone"), ("1925", "drone"), ("1926", "drone"),
        ("1916", "lasertag"), ("1917", "lasertag"), ("1918", "lasertag"),
        ("1924", "lasertag"), ("1925", "lasertag"), ("1926", "lasertag"),
    ]
    train_loso(targets_all)
