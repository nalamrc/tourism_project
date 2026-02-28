
import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

TARGET_COL = "ProdTaken"
HF_USERNAME = os.getenv("HF_USERNAME", "nalamrc")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", f"{HF_USERNAME}/tourism-wellness-dataset")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", f"{HF_USERNAME}/tourism-wellness-model")
HF_TOKEN = os.getenv("HF_TOKEN", "")
BASE_DIR = Path(os.getenv("GITHUB_SRC_ART_BASE_DIR", ".")).expanduser()

MODEL_DIR = BASE_DIR / "model_building"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"file:{(MODEL_DIR / 'mlruns').resolve()}")
mlflow.set_experiment("tourism_prod_experiments")

if HF_TOKEN and "nalamrc" not in HF_DATASET_REPO:
    train_file = hf_hub_download(repo_id=HF_DATASET_REPO, filename="train.csv", repo_type="dataset")
    test_file = hf_hub_download(repo_id=HF_DATASET_REPO, filename="test.csv", repo_type="dataset")
else:
    train_file = DATA_DIR / "train.csv"
    test_file = DATA_DIR / "test.csv"

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train = train_data.drop(columns=[TARGET_COL])
y_train = train_data[TARGET_COL].astype(int)
X_test = test_data.drop(columns=[TARGET_COL])
y_test = test_data[TARGET_COL].astype(int)

numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [col for col in X_train.columns if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ]
)

candidate_models = {
    "decision_tree": (
        DecisionTreeClassifier(random_state=42),
        {"model__max_depth": [4, 6, 8], "model__min_samples_split": [2, 8]},
    ),
    "bagging": (
        BaggingClassifier(random_state=42),
        {"model__n_estimators": [100, 200], "model__max_samples": [0.7, 1.0]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {"model__n_estimators": [200, 400], "model__max_depth": [None, 8]},
    ),
    "adaboost": (
        AdaBoostClassifier(random_state=42),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {"model__n_estimators": [150, 250], "model__learning_rate": [0.05, 0.1]},
    ),
}

best_score = -1.0
best_model_name = None
best_pipeline = None

for model_name, (model, param_grid) in candidate_models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring="f1", cv=3, n_jobs=-1)
    search.fit(X_train, y_train)

    estimator = search.best_estimator_
    preds = estimator.predict(X_test)
    probs = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator.named_steps["model"], "predict_proba") else preds

    metrics_payload = {
        "f1": f1_score(y_test, preds),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    with mlflow.start_run(run_name=f"prod_{model_name}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics_payload)

    if metrics_payload["f1"] > best_score:
        best_score = metrics_payload["f1"]
        best_model_name = model_name
        best_pipeline = estimator

model_path = MODEL_DIR / "best_model.joblib"
joblib.dump(best_pipeline, model_path)

metadata = {
    "best_model_name": best_model_name,
    "best_f1": float(best_score),
    "target": TARGET_COL,
    "features": X_train.columns.tolist(),
}
(MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
(MODEL_DIR / "README.md").write_text(
    "# Tourism Wellness Package Classifier" + chr(10) + chr(10)
    + "This model predicts whether a customer will purchase the wellness tourism package." + chr(10)
    + f"Best production model: {best_model_name}." + chr(10)
)

if HF_TOKEN and "nalamrc" not in HF_MODEL_REPO:
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=str(model_path), path_in_repo="best_model.joblib", repo_id=HF_MODEL_REPO, repo_type="model")
    api.upload_file(path_or_fileobj=str(MODEL_DIR / "README.md"), path_in_repo="README.md", repo_id=HF_MODEL_REPO, repo_type="model")
    api.upload_file(path_or_fileobj=str(MODEL_DIR / "model_metadata.json"), path_in_repo="model_metadata.json", repo_id=HF_MODEL_REPO, repo_type="model")

pd.DataFrame([metadata])
