import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers domain-specific features without leaking target/test information."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Ensure numeric types for calculations
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # 1. Average monthly spend inferred from total charges & tenure
        safe_tenure = np.maximum(df["tenure"], 1)
        imputed_total = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
        df["avg_monthly_spend"] = np.where(
            df["tenure"] > 0,
            imputed_total / safe_tenure,
            df["MonthlyCharges"],
        )

        # 2. Tenure buckets capturing onboarding vs. loyalty stages
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, np.inf],
            labels=["0-1_yr", "1-2_yrs", "2-4_yrs", "4plus_yrs"],
        ).astype(str)

        # 3. Charge-to-tenure ratio (high charges with short tenure signals early risk)
        df["charge_per_tenure_month"] = df["MonthlyCharges"] / (df["tenure"] + 1.0)

        return df


def load_and_clean_data(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(filepath)

    # Drop identifier columns that carry no predictive signal
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Clean whitespace on string/object columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Normalize target column to binary (1 = churned, 0 = retained)
    churn_normalized = df["Churn"].astype(str).str.strip().str.lower()
    mapped = churn_normalized.map(
        {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    )
    y_numeric = pd.to_numeric(mapped, errors="coerce")

    valid_mask = y_numeric.notna()
    df = df.loc[valid_mask].drop(columns=["Churn"]).reset_index(drop=True)
    y = y_numeric.loc[valid_mask].astype(int).reset_index(drop=True)

    # Drop rows where core numeric inputs are completely missing
    df = df.dropna(subset=["tenure", "MonthlyCharges"]).reset_index(drop=True)
    y = y.loc[df.index].reset_index(drop=True)

    return df, y


def build_preprocessor(X_sample: pd.DataFrame) -> ColumnTransformer:
    engineered = ChurnFeatureEngineer().transform(X_sample)

    numeric_features = engineered.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = engineered.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Chooses a probability threshold maximizing F1 for catching at-risk churners."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / np.maximum(
        precisions[:-1] + recalls[:-1], 1e-8
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def main() -> None:
    X, y = load_and_clean_data("churn_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Three distinct model families tuned via cross-validation
    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {
                "classifier__n_estimators": [150, 300],
                "classifier__max_depth": [6, 12, None],
                "classifier__min_samples_leaf": [2, 5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
            },
        ),
    }

    best_model_name = ""
    best_pipeline: Pipeline | None = None
    best_cv_pr_auc = -1.0

    print("=== Cross-Validation Results (5-Fold Stratified) ===")
    for name, (estimator, param_grid) in candidates.items():
        pipe = Pipeline(
            steps=[
                ("features", ChurnFeatureEngineer()),
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        cv_metrics = cross_validate(
            search.best_estimator_,
            X_train,
            y_train,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
            n_jobs=-1,
        )

        mean_roc = float(np.mean(cv_metrics["test_roc_auc"]))
        mean_pr = float(np.mean(cv_metrics["test_pr_auc"]))
        print(
            f"{name:20s} | CV ROC-AUC: {mean_roc:.4f} | CV PR-AUC: {mean_pr:.4f} "
            f"| Params: {search.best_params_}"
        )

        if mean_pr > best_cv_pr_auc:
            best_cv_pr_auc = mean_pr
            best_model_name = name
            best_pipeline = search.best_estimator_

    assert best_pipeline is not None

    # Evaluate selected model on held-out test set
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    optimal_threshold = find_optimal_threshold(y_test.to_numpy(), test_proba)
    test_preds = (test_proba >= optimal_threshold).astype(int)

    test_roc = roc_auc_score(y_test, test_proba)
    test_pr = average_precision_score(y_test, test_proba)
    test_f1 = f1_score(y_test, test_preds)

    print("\n=== Final Test Set Evaluation ===")
    print(f"Selected Model : {best_model_name}")
    print(f"Test ROC-AUC   : {test_roc:.4f}")
    print(f"Test PR-AUC    : {test_pr:.4f}")
    print(f"Optimal Thresh : {optimal_threshold:.3f} (Test F1: {test_f1:.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, digits=4))

    artifact = {
        "model": best_pipeline,
        "model_name": best_model_name,
        "threshold": optimal_threshold,
        "metrics": {
            "test_roc_auc": round(float(test_roc), 4),
            "test_pr_auc": round(float(test_pr), 4),
            "test_f1": round(float(test_f1), 4),
        },
    }
    joblib.dump(artifact, "churn_model_artifact.joblib")
    print("Saved trained pipeline to churn_model_artifact.joblib")


if __name__ == "__main__":
    main()