# Tomatometer Predictor: ML Assignment 2

Live app: **<STREAMLIT_APP_URL>**

## a. Problem statement

Rotten Tomatoes gives every movie a critics' verdict: **Rotten**, **Fresh** or **Certified-Fresh**. Can that verdict be predicted **without using any critic data**, from only the film's details (genre, content rating, runtime, release timing), who made it, and audience reaction?

This is a 3-class classification problem. It also tests how far audience opinion and film details agree with critics. Five classification models are trained and compared on the same data, and the trained models are served through an interactive Streamlit app that evaluates them on test data.

## b. Dataset description

| | |
|---|---|
| Source | [Rotten Tomatoes movies and critic reviews dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) (Kaggle, Stefano Leone, CC0 licence), file `rotten_tomatoes_movies.csv` |
| Instances | 17,712 movies, with 17,668 used after dropping 44 that have no verdict |
| Raw columns | 22 |
| Features used | 37 (derived from the raw columns, see below) |
| Target | `tomatometer_status`: Rotten, Fresh or Certified-Fresh |
| Train / test split | 80 / 20, stratified, `random_state=7`: 14,134 train, 3,534 test (`test_data.csv`) |

**Class distribution.** A model that always guessed "Rotten" would score 42.8% accuracy, so that's the baseline to beat.

| Rotten | Fresh | Certified-Fresh |
|---|---|---|
| 7,565 (42.8%) | 6,844 (38.7%) | 3,259 (18.4%) |

**Features (37).**
- **6 numeric:** `runtime`, `release_year`, `release_month`, `audience_rating`, `log_audience_count` (the raw count is extremely skewed), `streaming_lag_years` (years between release and streaming).
- **6 content rating flags:** `rated_G`, `rated_PG`, `rated_PG-13`, `rated_R`, `rated_NC17`, `rated_NR`.
- **21 genre flags:** e.g. `genre_Drama`, `genre_Comedy`, `genre_Horror`. A movie can have several.
- **4 people columns:** `director`, `studio`, `writer`, `lead_actor` (the first one credited). Each name is target-encoded into that person's past verdict rates, which gives every film a "track record". The encoder is fitted on the training data only and cross-fitted, so a film can't see its own label. Names not seen in training fall back to the overall rates.

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
├── test_data.csv          # held-out 20% test split (37 features, movie_title, target)
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

All 5 models use the same training split. Hyperparameters were chosen by 3-fold cross-validation on the **training data only**, using macro F1, so the test set played no part in tuning. Each saved model is a full pipeline, including imputation, target encoding and scaling where needed, so the app passes the raw columns straight in.

| Model | Configuration (best values from the grid search) |
|---|---|
| Logistic Regression | Preprocessing → StandardScaler → LogisticRegression (`C=10`) |
| Decision Tree | Preprocessing → DecisionTreeClassifier (`max_depth=10`, `min_samples_leaf=50`) |
| KNN | Preprocessing → StandardScaler → KNeighborsClassifier (`k=25`) |
| Naive Bayes | Preprocessing → GaussianNB |
| Support Vector Machine | Preprocessing → StandardScaler → RBF SVC (`C=1`), calibrated for probabilities |

Preprocessing is the same for all five: median imputation for the numeric columns and target encoding for the people columns.

### Comparison table (test set, 3,534 movies)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.7165** | **0.8701** | 0.7043 | **0.6928** | **0.6976** | **0.5489** |
| Decision Tree | 0.7009 | 0.8563 | 0.6899 | 0.6862 | 0.6879 | 0.5260 |
| KNN | 0.6517 | 0.8072 | 0.6477 | 0.5976 | 0.6088 | 0.4368 |
| Naive Bayes | 0.6126 | 0.7874 | 0.5991 | 0.5783 | 0.5767 | 0.3900 |
| Support Vector Machine | 0.7139 | 0.8477 | **0.7059** | 0.6912 | 0.6973 | 0.5443 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | **Best on 5 of 6 metrics** (Acc 0.717, AUC 0.870, F1 0.698, MCC 0.549), including the best ranking of the classes. It gained the most from the people features, going from 0.695 to 0.717 accuracy, because a track record is a simple numeric rate that suits a linear model. Its coefficients make sense: a higher audience rating pushes towards Certified-Fresh, and a director with a Rotten history pushes towards Rotten. Cross-validation and test scores agree (macro F1 0.686 vs 0.698), so there's no sign of overfitting. |
| Decision Tree | Third, but respectable (Acc 0.701, MCC 0.526), and the **best at finding Certified-Fresh movies** (recall 0.63). Feature importance is led by `audience_rating` (48%) and `log_audience_count` (18%), with the studio's track record at 10%. It gained little from the people features (0.698 to 0.701), since a depth-limited tree can only use so many splits. Its accuracy is significantly below Logistic Regression (McNemar χ² = 4.92). It is the easiest model to explain. |
| KNN | Weak (Acc 0.652, MCC 0.437), and the only model that got worse with the people features (0.658 to 0.652). 27 of the 33 non-people features are binary flags, and encoding adds 12 more columns, so distances are probably noisy. It rarely predicts Certified-Fresh (11% of movies vs 18% actual), which gives it the lowest recall for that class (0.37). |
| Naive Bayes | Still the lowest (Acc 0.613, MCC 0.390), though above the 42.8% baseline, and it improved by 3.4 points with the people features. GaussianNB assumes each feature is normally distributed and independent, but most features are 0/1 flags, and the three rates encoded for each person are directly related. It predicts Rotten for 57% of movies, giving high Rotten recall (0.82) but low Fresh recall (0.46). |
| Support Vector Machine | A very close second (Acc 0.714, F1 0.697) with the **highest Precision (0.706)**. Its accuracy is not significantly different from Logistic Regression (McNemar χ² = 0.18). Its AUC (0.848) is below both Logistic Regression and the Decision Tree, so its probabilities are less useful than its hard predictions. It is also the slowest model to train. |
| **Overall winner** | **Logistic Regression**, on 5 of 6 metrics. Its lead over the SVM is within the margin of error (about ±1.5 points of accuracy), but it is also far simpler and faster, so it is the clear practical choice. |

**Main finding.** Film details, track records and audience reaction reach about 72% accuracy, well above the 42.8% baseline, using no critic data. Rotten is the easiest class to predict. For Logistic Regression, 84% of errors (845 of 1,002) are between neighbouring classes (Fresh vs Rotten, Fresh vs Certified-Fresh), and the two extremes are confused least often. The app's "Biggest misses" tab shows real examples: *Hulk* has an audience score of only 29 but is Fresh with critics, and the model called it Rotten with 97% confidence.

**What didn't help.** Scored by cross-validation on the training data, adding the plot synopsis (TF-IDF with SVD) didn't raise accuracy. Neither did more cast members or counts of credited people.

**Limitations.** Many test films have people the model has never seen (34% of directors, 35% of lead actors, 45% of writers and 13% of studios), so they get no real track record. Track records also use all training films, including ones released after a given test film, so this measures how well verdicts can be explained rather than how well they could be forecast. Finally, Certified-Fresh partly depends on how many critics reviewed a film, which is deliberately hidden here. That likely caps further gains.

## Streamlit app features

- **CSV upload** of test data (a sample `test_data.csv` can be downloaded from the sidebar). The bundled test data is shown by default, so results are visible immediately.
- **Model selection dropdown** for all 5 saved models
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix** heatmap and **classification report**
- **Compare all models** tab: all 5 models scored side by side
- **Biggest misses** tab: the movies the selected model was most confidently wrong about
- Download the predictions as CSV
- The app only loads the `.joblib` models; it never retrains on the uploaded data.
