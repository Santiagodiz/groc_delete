import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    zero_one_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Must declare data_dir as the directory of training and test files
# data_dir = "./datasets/CIC/"
SCRIPT_DIR = Path(__file__).resolve().parent
data_dir = SCRIPT_DIR.parent
# raw_data_filename = data_dir / "Obfuscated-MalMem2022_labeled.10_percent.csv"
raw_data_filename = data_dir / "Obfuscated-MalMem2022_labeled.csv"
RANDOM_STATE = 42


def load_raw_data(csv_path):
    print("Loading raw data")
    try:
        return pd.read_csv(csv_path, header=None)
    except pd.errors.ParserError as exc:
        # Some rows in the full dataset can be malformed (extra separator tokens).
        print(f"Standard parser failed: {exc}")
        print("Retrying with robust parser (auto-separator, skipping malformed rows)...")

        raw = pd.read_csv(
            csv_path,
            header=None,
            sep=None,
            engine="python",
            on_bad_lines="skip",
        )

        with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
            total_lines = sum(1 for _ in fh)

        skipped_rows = max(total_lines - raw.shape[0], 0)
        print(f"Loaded rows: {raw.shape[0]} | Approx skipped malformed rows: {skipped_rows}")

        if raw.empty:
            raise RuntimeError(
                f"No valid rows could be parsed from dataset: {Path(csv_path).name}"
            )

        return raw


raw_data = load_raw_data(raw_data_filename)

print("Transforming data")
# Column 0 has specific malware name, ignore it.
# Last column has malware class (Benign, Spyware, Ransomware, Trojan).
# Last but one column has malware type.
labels = raw_data.iloc[:, raw_data.shape[1] - 1 :]
features = raw_data.iloc[:, 1 : raw_data.shape[1] - 2]
labels = labels.values.ravel()

# Force numerical types for model safety.
features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

print("labels:", np.unique(labels), "\n")

# Create training and testing vars.
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    train_size=0.8,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=labels,
)

print("X_train, y_train:", X_train.shape, y_train.shape)
print("X_test, y_test:", X_test.shape, y_test.shape)

all_classes = np.unique(labels)


def compute_multiclass_auc(model, X_test_data, y_test_data, classes):
    y_test_bin = label_binarize(y_test_data, classes=classes)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_data)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_data)
        if y_score.ndim == 1:
            y_score = np.column_stack([-y_score, y_score])
    else:
        return np.nan

    try:
        return roc_auc_score(
            y_test_bin,
            y_score,
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        return np.nan


def evaluate_supervised(name, model, X_tr, y_tr, X_te, y_te, classes):
    trained = clone(model).fit(X_tr, y_tr)
    y_pred = trained.predict(X_te)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_te, y_pred),
        "zero_one_loss": zero_one_loss(y_te, y_pred),
        "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "auc_ovr_macro": compute_multiclass_auc(trained, X_te, y_te, classes),
    }

    print("\n" + "=" * 90)
    print(f"MODEL: {name}")
    print("=" * 90)
    print("Confusion matrix:\n", confusion_matrix(y_te, y_pred, labels=classes))
    print("Classification report:\n", classification_report(y_te, y_pred, zero_division=0))

    return metrics


def map_clusters_to_labels(y_train_data, clusters_train):
    cluster_to_class = {}
    for cluster_id in np.unique(clusters_train):
        mask = clusters_train == cluster_id
        labels_in_cluster = y_train_data[mask]
        values, counts = np.unique(labels_in_cluster, return_counts=True)
        cluster_to_class[cluster_id] = values[np.argmax(counts)]
    return cluster_to_class


def evaluate_kmeans(X_tr, y_tr, X_te, y_te, classes):
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    kmeans = KMeans(
        n_clusters=len(classes),
        random_state=RANDOM_STATE,
        n_init=10,
    )

    clusters_train = kmeans.fit_predict(X_tr_scaled)
    clusters_test = kmeans.predict(X_te_scaled)

    cluster_to_class = map_clusters_to_labels(y_tr, clusters_train)
    default_class = pd.Series(y_tr).mode().iloc[0]
    y_pred = np.array([cluster_to_class.get(c, default_class) for c in clusters_test])

    metrics = {
        "model": "KMeans (cluster->class)",
        "accuracy": accuracy_score(y_te, y_pred),
        "zero_one_loss": zero_one_loss(y_te, y_pred),
        "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "auc_ovr_macro": np.nan,
    }

    print("\n" + "=" * 90)
    print("MODEL: KMeans (cluster->class)")
    print("=" * 90)
    print("Confusion matrix:\n", confusion_matrix(y_te, y_pred, labels=classes))
    print("Classification report:\n", classification_report(y_te, y_pred, zero_division=0))

    return metrics


print("\nTraining and evaluating models")

models = [
    (
        "DecisionTree",
        DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "RandomForest",
        RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "SVC (RBF)",
        make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
        ),
    ),
    (
        "KNearestNeighbor",
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
    ),
]

results = []
for model_name, model in models:
    results.append(
        evaluate_supervised(
            model_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            all_classes,
        )
    )

results.append(evaluate_kmeans(X_train, y_train, X_test, y_test, all_classes))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["f1_macro", "accuracy"], ascending=False)

print("\n" + "#" * 90)
print("RESUMEN COMPARATIVO DE PRESTACIONES")
print("#" * 90)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best_row = results_df.iloc[0]
print("\nMejor modelo por F1-macro:")
print(
    f"- {best_row['model']} | "
    f"Accuracy={best_row['accuracy']:.4f}, "
    f"F1-macro={best_row['f1_macro']:.4f}, "
    f"AUC-OVR-macro={best_row['auc_ovr_macro']:.4f}"
)

print("\nNota: para KMeans no se calcula AUC de clasificación supervisada.")
