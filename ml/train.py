"""
Train and compare milk-quality classifiers, then save the best one.

Run from the project root:   python ml/train.py

Why two scores?
The dataset has 1,059 rows but only 83 unique samples (976 rows are exact
duplicates). A normal random split puts copies of the same sample in both the
training and test sets, so accuracy looks higher than it really is. We report:

  * standard accuracy  - stratified 10-fold CV on all rows (comparable to the
                          original notebook)
  * unseen accuracy    - grouped CV where every copy of a sample stays in the
                          same fold, so the model is always tested on samples it
                          has never seen. This is the honest number.
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "milknew.csv"
OUT_MODEL = ROOT / "ml" / "milk_model.joblib"
OUT_METRICS = ROOT / "ml" / "metrics.json"

FEATURES = ["pH", "Temperature", "Taste", "Odor", "Fat", "Turbidity", "Colour"]
CLASSES = ["low", "medium", "high"]  # encoded 0, 1, 2
SEED = 42


def load():
    df = pd.read_csv(DATA)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Temprature": "Temperature"})
    df["Grade"] = df["Grade"].str.strip().str.lower().map({c: i for i, c in enumerate(CLASSES)})
    X, y = df[FEATURES].astype(float), df["Grade"].astype(int)
    groups = X.astype(str).agg("|".join, axis=1).factorize()[0]  # one id per unique sample
    return df, X, y, groups


def candidates():
    """The original project's five algorithms (scaled where they need it) plus stronger ensembles.
    Settings are fixed rather than grid-searched on the evaluation folds, so scores aren't inflated."""
    trees = lambda: [
        ("hgb", HistGradientBoostingClassifier(random_state=SEED, min_samples_leaf=5)),
        ("gb", GradientBoostingClassifier(random_state=SEED, max_depth=2, n_estimators=300)),
        ("rf", RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1)),
        ("et", ExtraTreesClassifier(n_estimators=400, random_state=SEED, n_jobs=-1)),
    ]
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=5000)),
        "SVM": make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=SEED)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=400, criterion="entropy", random_state=SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=400, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, max_depth=2, random_state=SEED),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(min_samples_leaf=5, random_state=SEED),
        "Soft Voting Ensemble": VotingClassifier(trees(), voting="soft"),
        "Stacked Ensemble": StackingClassifier(trees(), final_estimator=LogisticRegression(max_iter=5000)),
    }


def scores(y_true, y_pred, proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "roc_auc": round(roc_auc_score(y_true, proba, multi_class="ovr") * 100, 2),
    }


def main():
    t0 = time.time()
    df, X, y, groups = load()
    std_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    grp_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, models = [], candidates()
    seeds = [42, 7, 123]  # repeat the honest split 3 times; with 83 unique samples one split is noisy
    for name, model in models.items():
        p_std = cross_val_predict(model, X, y, cv=std_cv, method="predict_proba", n_jobs=-1)
        unseen = []
        for s in seeds:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
            p = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba", n_jobs=-1)
            unseen.append(scores(y, p.argmax(1), p))
        x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        row = {
            "name": name,
            "standard": scores(y, p_std.argmax(1), p_std),
            "unseen": {k: round(float(np.mean([u[k] for u in unseen])), 2) for k in unseen[0]},
            "unseen_spread": round(float(np.std([u["accuracy"] for u in unseen])), 2),
            "holdout": round(clone(model).fit(x_tr, y_tr).score(x_te, y_te) * 100, 2),
        }
        results.append(row)
        print(f"{name:24s} holdout {row['holdout']:6.2f}%  standard {row['standard']['accuracy']:6.2f}%  unseen {row['unseen']['accuracy']:6.2f}%")

    # Pick the best: among models within 2 points of the top unseen accuracy (the
    # run-to-run noise on 83 unique samples), take the highest 80/20 holdout score.
    top = max(r["unseen"]["accuracy"] for r in results)
    results.sort(key=lambda r: (r["unseen"]["accuracy"] >= top - 2, r["holdout"], r["unseen"]["accuracy"]), reverse=True)
    best_name = results[0]["name"]
    model = models[best_name]
    holdout_acc = results[0]["holdout"]
    for r in results:
        r["best"] = r["name"] == best_name

    # Confusion matrix + per-class report from honest grouped CV predictions.
    p_grp = cross_val_predict(model, X, y, cv=grp_cv, groups=groups, n_jobs=-1)
    cm = confusion_matrix(y, p_grp, labels=[0, 1, 2]).tolist()
    report = classification_report(y, p_grp, target_names=CLASSES, output_dict=True, zero_division=0)

    model.fit(X, y)  # final model on all data
    # Model-agnostic feature importance: how much accuracy drops when a feature is shuffled.
    pi = permutation_importance(model, X, y, n_repeats=15, random_state=SEED, n_jobs=-1)
    imp = np.clip(pi.importances_mean, 0, None)
    imp = imp / imp.sum() if imp.sum() else imp
    importances = sorted(({"feature": f, "value": round(float(v) * 100, 2)} for f, v in zip(FEATURES, imp)),
                         key=lambda d: -d["value"])

    joblib.dump({"model": model, "features": FEATURES, "classes": CLASSES, "name": best_name}, OUT_MODEL)

    metrics = {
        "best_model": best_name,
        "holdout_accuracy": holdout_acc,
        "original_best": {"name": "Random Forest", "accuracy": 99.53},
        "dataset": {
            "rows": int(len(df)), "unique_samples": int(groups.max() + 1),
            "duplicates": int(len(df) - groups.max() - 1),
            "class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
            "ranges": {f: [float(X[f].min()), float(X[f].max())] for f in FEATURES},
        },
        "models": results,
        "confusion_matrix": cm,
        "per_class": {c: {k: round(report[c][k] * 100, 1) for k in ("precision", "recall", "f1-score")} for c in CLASSES},
        "feature_importance": importances,
        "trained_seconds": round(time.time() - t0, 1),
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"\nBest: {best_name}  |  80/20 holdout {holdout_acc}%  |  saved {OUT_MODEL.name}, {OUT_METRICS.name}")


if __name__ == "__main__":
    main()
