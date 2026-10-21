"""Optional inexpensive baseline; train and dev remain explicitly separate."""

from pathlib import Path

from .data import write_json, write_jsonl
from .evaluation import assert_no_overlap, evaluate, index_gold


def train_baseline(train, dev, output_dir, seed=42, folds=5):
    index_gold(train)
    index_gold(dev)
    assert_no_overlap(train, dev)
    labels = [row["Output"]["Error_Type_Id"] for row in train]
    if len(set(labels)) < 2:
        raise ValueError("A classifier needs multiple labels; combine label-2 augmentation with a real multi-class train set")
    from collections import Counter
    if folds < 2 or min(Counter(labels).values()) < folds:
        raise ValueError("Each training label needs at least folds examples for out-of-fold hard mining")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Classifier output directory must be empty")
    try:
        import joblib
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise RuntimeError('Install classifier extras: python -m pip install -e ".[classifier]"') from exc
    # Docs: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    # Character n-grams avoid requiring a Vietnamese word segmentation dependency.
    model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=100_000),
                          LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    texts = [row["Input"] for row in train]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(model, texts, labels, cv=cv, method="predict_proba")
    classes = sorted(set(labels))

    def predictions(rows, probs):
        return [{"ID": row["ID"], "label": int(classes[int(vector.argmax())]),
                 "confidence": float(vector.max())} for row, vector in zip(rows, probs)]

    oof = predictions(train, probabilities)
    model.fit(texts, labels)
    dev_predictions = predictions(dev, model.predict_proba([row["Input"] for row in dev]))
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, root / "baseline.joblib")
    write_jsonl(root / "train_oof_predictions.jsonl", oof)
    write_jsonl(root / "dev_predictions.jsonl", dev_predictions)
    report = evaluate(dev, dev_predictions)
    report["training"] = {"seed": seed, "folds": folds, "sklearn_version": sklearn.__version__,
                          "train_count": len(train), "classes": classes, "hard_mining_predictions": "out_of_fold"}
    write_json(root / "dev_metrics.json", report)
    return report
