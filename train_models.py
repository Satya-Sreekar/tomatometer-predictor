"""Train the 5 classifiers on the Rotten Tomatoes movies dataset and save them to model/.

Goal: predict the critics' verdict (Rotten / Fresh / Certified-Fresh) from film
details and audience reaction. Critic-side columns are excluded on purpose, see
build_features().

Run:  python train_models.py
Outputs: model/*.joblib, model/metrics.csv, model/best_params.csv, test_data.csv
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics import evaluate

SEED = 7
TARGET = "tomatometer_status"
PEOPLE = ["director", "studio", "writer", "lead_actor"]


def build_features(raw):
    """Turn the raw columns into one flat table: numeric features plus PEOPLE names.

    Left out on purpose: tomatometer_rating and the fresh/rotten critic counts
    define the label (they push accuracy to ~98%), tomatometer_count and the top
    critic count feed the Certified-Fresh rule, and critics_consensus is written
    by critics. audience_status is just audience_rating >= 60, so it adds nothing.
    """
    released = pd.to_datetime(raw["original_release_date"], errors="coerce")
    streamed = pd.to_datetime(raw["streaming_release_date"], errors="coerce")
    numeric = pd.DataFrame({
        "runtime": raw["runtime"],
        "release_year": released.dt.year,
        "release_month": released.dt.month,
        "audience_rating": raw["audience_rating"],
        "log_audience_count": np.log1p(raw["audience_count"]),
        "streaming_lag_years": (streamed - released).dt.days / 365.25,
    })
    rated = pd.get_dummies(raw["content_rating"], prefix="rated", dtype=int)
    genres = raw["genres"].fillna("").str.get_dummies(", ").add_prefix("genre_")
    first_name = lambda col: raw[col].fillna("Unknown").str.split(", ").str[0]
    people = pd.DataFrame({
        "director": first_name("directors"),
        "studio": raw["production_company"].fillna("Unknown"),
        "writer": first_name("authors"),
        "lead_actor": first_name("actors"),
    })
    genres = genres.drop(columns="genre_", errors="ignore")
    return pd.concat([numeric, rated, genres, people], axis=1)


raw = pd.read_csv("data/rotten_tomatoes_movies.csv").dropna(subset=[TARGET])
X, y = build_features(raw), raw[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)


def pipe(*steps):
    # Preprocessing lives inside the pipeline, so medians and track records come from
    # the training data only. TargetEncoder turns each name into that person's past
    # verdict rates, cross-fitting on the training rows so a film can't see its own
    # label. Names not seen in training fall back to the overall rates.
    numeric = [c for c in X.columns if c not in PEOPLE]
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("people", TargetEncoder(target_type="multiclass", random_state=SEED), PEOPLE),
    ])
    return make_pipeline(preprocess, *steps)


# (estimator, grid). Grids are searched with CV on the training split only.
models = {
    "Logistic Regression": (
        pipe(StandardScaler(), LogisticRegression(max_iter=3000)),
        {"logisticregression__C": [1, 10, 100]},
    ),
    "Decision Tree": (
        pipe(DecisionTreeClassifier(random_state=SEED)),
        {"decisiontreeclassifier__max_depth": [6, 8, 10, 14],
         "decisiontreeclassifier__min_samples_leaf": [10, 25, 50, 100]},
    ),
    "KNN": (
        pipe(StandardScaler(), KNeighborsClassifier()),
        {"kneighborsclassifier__n_neighbors": [9, 15, 25, 41, 61]},
    ),
    "Naive Bayes": (pipe(GaussianNB()), {}),
    # sklearn 1.9 deprecates SVC(probability=True); calibration gives predict_proba for AUC.
    "Support Vector Machine": (
        pipe(StandardScaler(), CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False)),
        {"calibratedclassifiercv__estimator__C": [0.1, 0.3, 1, 3]},
    ),
}

rows, params = [], []
for name, (estimator, grid) in models.items():
    search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    joblib.dump(model, f"model/{name.lower().replace(' ', '_')}.joblib", compress=3)
    rows.append({"Model": name, **evaluate(model, X_test, y_test)})
    params.append({"Model": name, "Best params": search.best_params_,
                   "CV macro F1": round(search.best_score_, 4)})
    print(f"trained {name}: {search.best_params_}", flush=True)

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv("model/metrics.csv", index=False)
pd.DataFrame(params).to_csv("model/best_params.csv", index=False)
# movie_title is not a feature; it only lets the app name the movies it gets wrong.
X_test.assign(movie_title=raw.loc[X_test.index, "movie_title"], **{TARGET: y_test}).to_csv(
    "test_data.csv", index=False
)
print(metrics.to_string(index=False))
