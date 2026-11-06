"""Tomatometer Predictor: Streamlit demo for ML Assignment 2.

Loads the 5 pre-trained models from model/ and evaluates them on an uploaded
test CSV (or the bundled test_data.csv). No retraining happens here.
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics import evaluate

TARGET = "tomatometer_status"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}

st.set_page_config(page_title="Tomatometer Predictor", page_icon="🍅", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(f"model/{f}.joblib") for name, f in MODEL_FILES.items()}


@st.cache_data
def score_all(X, y):
    return pd.DataFrame({name: evaluate(m, X, y) for name, m in models.items()}).T


models = load_models()
features = list(models["Naive Bayes"].feature_names_in_)

# ---------- sidebar ----------
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader(f"Upload test CSV (features + '{TARGET}')", type="csv")
    with open("test_data.csv", "rb") as f:
        st.download_button("Download sample test_data.csv", f, "test_data.csv", "text/csv")
    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
    st.caption("Models were trained offline (train_models.py) on 80% of the Rotten Tomatoes "
               "movies dataset. test_data.csv is the held-out 20%.")

st.title("🍅 Tomatometer Predictor")
st.write("Can you tell what critics will think of a movie without reading a single review? "
         "These models predict **Rotten**, **Fresh** or **Certified-Fresh** from film details "
         "and audience reaction only. All critic data is hidden from them.")

df = pd.read_csv(uploaded if uploaded else "test_data.csv")
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing columns: {missing}")
    st.stop()
df = df.dropna(subset=[TARGET])  # missing feature values are imputed by the saved pipelines
X, y = df[features], df[TARGET]
source = uploaded.name if uploaded else "the bundled test_data.csv"
st.success(f"Evaluating on {source}: {len(df):,} movies across {y.nunique()} classes.")

table = score_all(X, y)
model = models[model_name]
pred = model.predict(X)

tab_model, tab_compare, tab_misses, tab_data = st.tabs(
    [f"📊 {model_name}", "🏆 Compare all models", "🎬 Biggest misses", "🔎 Data preview"]
)

# ---------- selected model ----------
with tab_model:
    for col, (k, v) in zip(st.columns(3) * 2, table.loc[model_name].items()):
        # AUC is undefined when the upload does not contain all 3 classes.
        col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Confusion matrix")
        labels = model.classes_
        cm = confusion_matrix(y, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y, pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    st.download_button(
        "Download predictions CSV",
        df.assign(Predicted=pred).to_csv(index=False),
        f"predictions_{MODEL_FILES[model_name]}.csv", "text/csv",
    )

# ---------- all models side by side ----------
with tab_compare:
    st.dataframe(
        table.style.format("{:.4f}", na_rep="n/a").highlight_max(axis=0, color="#f7b5a8"),
        width="stretch",
    )
    st.bar_chart(table[["Accuracy", "F1", "MCC"]], stack=False)
    st.caption(f"Best by MCC on this data: **{table['MCC'].idxmax()}**")

# ---------- movies the selected model got most wrong ----------
with tab_misses:
    proba = pd.DataFrame(model.predict_proba(X), columns=model.classes_, index=df.index)
    wrong = df[pred != y.to_numpy()]
    if wrong.empty:
        st.write("No misclassified movies.")
    else:
        confidence = proba.loc[wrong.index].max(axis=1)
        st.write(f"{model_name} got **{len(wrong):,}** of {len(df):,} movies wrong. "
                 "These are the ones it was most confident about:")
        cols = ["movie_title"] if "movie_title" in df else []
        worst = wrong[cols + [TARGET, "audience_rating"]].assign(
            Predicted=pd.Series(pred, index=df.index).loc[wrong.index],
            Confidence=confidence.round(3),
        )
        st.dataframe(worst.sort_values("Confidence", ascending=False).head(15),
                     width="stretch", hide_index=True)

with tab_data:
    st.dataframe(df.head(50), width="stretch")
    st.bar_chart(y.value_counts())
