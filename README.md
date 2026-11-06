# Tomatometer Predictor: ML Assignment 2

Live app: **<STREAMLIT_APP_URL>**

## a. Problem statement

Rotten Tomatoes gives every movie a critics' verdict: **Rotten**, **Fresh** or **Certified-Fresh**. Can that verdict be predicted **without using any critic data**, from only the film's details (genre, content rating, runtime, release timing) and audience reaction?

This is a 3-class classification problem. It also tests how far audience opinion and film details agree with critics. Five classification models are trained and compared on the same data, and the trained models are served through an interactive Streamlit app that evaluates them on test data.

## b. Dataset description

| | |
|---|---|
| Source | [Rotten Tomatoes movies and critic reviews dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) (Kaggle, Stefano Leone, CC0 licence), file `rotten_tomatoes_movies.csv` |
| Instances | 17,712 movies, with 17,668 used after dropping 44 that have no verdict |
| Raw columns | 22 |
| Features used | 33 (derived from the raw columns, see below) |
| Target | `tomatometer_status`: Rotten, Fresh or Certified-Fresh |
| Train / test split | 80 / 20, stratified, `random_state=7`: 14,134 train, 3,534 test (`test_data.csv`) |

**Class distribution.** A model that always guessed "Rotten" would score 42.8% accuracy, so that's the baseline to beat.

| Rotten | Fresh | Certified-Fresh |
|---|---|---|
| 7,565 (42.8%) | 6,844 (38.7%) | 3,259 (18.4%) |

**Features (33).**
- **6 numeric:** `runtime`, `release_year`, `release_month`, `audience_rating`, `log_audience_count` (the raw count is extremely skewed), `streaming_lag_years` (years between release and streaming).
- **6 content rating flags:** `rated_G`, `rated_PG`, `rated_PG-13`, `rated_R`, `rated_NC17`, `rated_NR`.
- **21 genre flags:** e.g. `genre_Drama`, `genre_Comedy`, `genre_Horror`. A movie can have several.

Missing values (for example, 6.6% of release dates) are filled with the training-set median inside each saved pipeline.

**Columns excluded to prevent leakage.** This was the most important preprocessing decision:

| Excluded column(s) | Why |
|---|---|
| `tomatometer_rating`, `tomatometer_fresh_critics_count`, `tomatometer_rotten_critics_count` | The label is calculated from these |
| `tomatometer_count`, `tomatometer_top_critics_count` | Critic data, and part of the Certified-Fresh rule |
| `critics_consensus` | Written by critics after reviewing |
| `audience_status` | Just a thresholded copy of `audience_rating` |

In a quick check with a Decision Tree, keeping the first group pushed accuracy to about 98%, and keeping only the counts gave about 75%. Those scores would look better but wouldn't answer the question. Without any critic data the same tree scored about 70%, so the results below are lower but honest.

Precision, Recall and F1 are reported as **macro averages**, so the smaller Certified-Fresh class counts equally. AUC is the macro-averaged **one-vs-rest** ROC AUC.

## c. GitHub repository link

**https://github.com/Satya-Sreekar/tomatometer-predictor**

```
├── app.py                 # Streamlit app (loads saved models, never retrains)
├── train_models.py        # feature engineering, tuning, training; writes model/ + test_data.csv
├── metrics.py             # the 6 evaluation metrics (shared by training and app)
├── requirements.txt
├── README.md
├── test_data.csv          # held-out 20% test split (33 features, movie_title, target)
├── .streamlit/config.toml # app theme
├── data/rotten_tomatoes_movies.csv
└── model/                 # 5 saved pipelines (.joblib), metrics.csv, best_params.csv
```

Run locally:

```bash
pip install -r requirements.txt
python train_models.py      # optional: regenerates model/ and test_data.csv (about 2 minutes)
streamlit run app.py
```

## d. Models used

All 5 models use the same training split. Hyperparameters were chosen by 3-fold cross-validation on the **training data only**, using macro F1, so the test set played no part in tuning. Each saved model is a full pipeline, including imputation and scaling where needed.

| Model | Configuration (best values from the grid search) |
|---|---|
| Logistic Regression | Median imputer → StandardScaler → LogisticRegression (`C=100`) |
| Decision Tree | Median imputer → DecisionTreeClassifier (`max_depth=10`, `min_samples_leaf=50`) |
| KNN | Median imputer → StandardScaler → KNeighborsClassifier (`k=15`) |
| Naive Bayes | Median imputer → GaussianNB |
| Support Vector Machine | Median imputer → StandardScaler → RBF SVC (`C=3`), calibrated for probabilities |

### Comparison table (test set, 3,534 movies)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6952 | **0.8542** | 0.6822 | 0.6647 | 0.6716 | 0.5134 |
| Decision Tree | 0.6984 | 0.8520 | 0.6844 | 0.6783 | 0.6809 | 0.5213 |
| KNN | 0.6582 | 0.8107 | 0.6433 | 0.6175 | 0.6261 | 0.4514 |
| Naive Bayes | 0.5787 | 0.7640 | 0.5636 | 0.5289 | 0.5249 | 0.3353 |
| Support Vector Machine | **0.7026** | 0.8309 | **0.6941** | **0.6790** | **0.6848** | **0.5264** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong for such a simple model (Acc 0.695), with the **highest AUC (0.854)**, so it ranks the classes best. Train and test accuracy are almost equal (0.690 vs 0.695), so it isn't overfitting, and regularisation made almost no difference. A linear model mostly captures "higher audience rating means better verdict". It finds Certified-Fresh movies less often (recall 0.54) than the tree or SVM. |
| Decision Tree | A very close second (Acc 0.698, MCC 0.521) and the **best at finding Certified-Fresh movies** (recall 0.60). Tuning chose a fairly constrained tree (`min_samples_leaf=50`), which keeps overfitting small (0.715 train vs 0.698 test). Feature importance is dominated by `audience_rating` (52%) and `log_audience_count` (21%). It handles binary flags and unscaled numbers naturally and is easy to explain. |
| KNN | Weaker than the top three (Acc 0.658, MCC 0.451), with a 3.5-point train/test gap (0.694 vs 0.658). 27 of the 33 features are binary flags, so distances are probably dominated by rare genres after scaling, which adds noise. It struggles most with Certified-Fresh (recall 0.45). |
| Naive Bayes | Clearly the worst, though still above the 42.8% baseline (Acc 0.579, MCC 0.335). GaussianNB assumes each feature is normally distributed and independent, but 27 features are 0/1 flags, and genres and ratings are related. It predicts Rotten for 62% of movies, giving high Rotten recall (0.84) but poor Fresh recall (0.41). The train and test scores are the same, so its limit is bias rather than overfitting. |
| Support Vector Machine | **Best on 5 of 6 metrics** (Acc 0.703, F1 0.685, MCC 0.526). The RBF kernel can model non-linear interactions between audience score, popularity and release year. It also has the biggest train/test gap (0.755 vs 0.703), and its AUC (0.831) is below Logistic Regression and the tree, so its probabilities are less useful than its hard predictions. It is also the slowest model to train. |
| **Overall winner** | **Support Vector Machine** on 5 of 6 metrics, but only narrowly. The accuracy margin of error is about ±1.5 points, and a McNemar test against the Decision Tree shows no significant difference (χ² = 0.34). SVM, Decision Tree and Logistic Regression are effectively tied, so the Decision Tree is the practical choice: it is faster, simpler and explainable. |

**Main finding.** Film details and audience reaction get about 70% of the way there, well above the 42.8% baseline, but not much further with any of these models, because audiences and critics do disagree. Rotten is the easiest class to predict. For the SVM, 84% of errors (882 of 1,051) are between neighbouring classes (Fresh vs Rotten, Fresh vs Certified-Fresh), and the two extremes are confused least often. The app's "Biggest misses" tab shows real examples: *Spy Kids* has an audience score of only 46 but is Certified-Fresh with critics.

## Streamlit app features

- **CSV upload** of test data (a sample `test_data.csv` can be downloaded from the sidebar). The bundled test data is shown by default, so results are visible immediately.
- **Model selection dropdown** for all 5 saved models
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix** heatmap and **classification report**
- **Compare all models** tab: all 5 models scored side by side
- **Biggest misses** tab: the movies the selected model was most confidently wrong about
- Download the predictions as CSV
- The app only loads the `.joblib` models; it never retrains on the uploaded data.
