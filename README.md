# Heart Disease Health Indicators (BRFSS 2015) — CP620 Project Dataset

This repository contains the datasets used in a CP620 Data Mining Programming project comparing three classifier families implemented from scratch:

- **Naïve Bayes (NB)** — generative classifier
- **Multilayer Neural Network (MNN)** — discriminative classifier trained by backpropagation
- **Decision Tree (DT)** — tree-based classifier trained with information-gain splitting

The project is a machine-learning classification study on self-reported survey indicators. It is **not** a clinical diagnosis or medical screening tool.

## Files

| File | Rows | Class balance | Purpose |
|---|---:|---:|---|
| **`heart_disease_balanced.csv`** | **10,000** | **50/50** (5,000 positive / 5,000 negative) | **Primary dataset** for controlled model-comparison experiments. |
| `heart_disease_clean.csv` | 15,000 | Stratified, approximately 9.42% positive | **Secondary imbalanced dataset** for separate 5-fold CV analysis under natural class prevalence. |
| `brfss_raw.csv` | 253,680 | Approximately 9.42% positive | Cleaned BRFSS 2015 source dataset used to regenerate both derived samples. |
| `preprocess.py` | — | — | Deterministic pipeline for generating the two disjoint derived datasets with seed 42. |

## Source and license

This project uses the **Heart Disease Health Indicators Dataset**, released on Kaggle by Alex Teboul from cleaned **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015** data.

- Original survey documentation: [CDC — 2015 BRFSS Survey Data and Documentation](https://www.cdc.gov/brfss/annual_data/annual_2015.html)
- Cleaned dataset release: [Alex Teboul — Heart Disease Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)
- Dataset license on Kaggle: **CC0: Public Domain**
- Cleaned dataset size used here: **253,680 responses and 22 columns**

## Target variable

`target` is binary:

- `0` — the respondent did not report being told by a doctor or health professional that they had coronary heart disease or a myocardial infarction.
- `1` — the respondent reported such a history.

This is a **self-reported history label**, not a prospective disease outcome and not a clinical diagnosis.

## Disjoint sampling guarantee

The two derived subsamples are generated to share **zero source-row indices**. The deterministic sampling procedure is:

1. Draw the 15,000-row stratified imbalanced sample first, preserving the source dataset's approximately 9.42% positive rate.
2. Remove those source rows from the available sampling pool.
3. Draw 5,000 positive and 5,000 negative rows from the remaining pool to create the balanced sample.
4. Verify in `preprocess.py` that the two sets of source-row indices are disjoint before saving the CSV outputs.

Source-row indices are used only to verify sampling integrity and are **not** model features.

BRFSS rows do not include a respondent identifier in this cleaned release. Therefore, two respondents may have identical feature values. Identical-content rows are retained; disjointness is defined at the original source-row index level.

## Two experimental regimes

The two samples are analysed separately:

| Regime | Dataset | Evaluation purpose |
|---|---|---|
| Balanced regime | `heart_disease_balanced.csv` | Controlled comparison of NB, MNN, and DT using stratified 5-fold cross-validation, confusion matrices, accuracy, precision, recall, F1, FPR, ROC curves, and ROC-AUC. |
| Imbalanced regime | `heart_disease_clean.csv` | Separate stratified 5-fold cross-validation under natural class prevalence, emphasizing positive-class retrieval through Precision-Recall curves and Average Precision (AP), reported alongside ROC-AUC. |

The imbalanced dataset illustrates why accuracy alone is insufficient: a no-skill classifier that predicts only the negative class can achieve approximately 90.6% accuracy while producing positive-class F1 equal to 0.

## Features (21 predictors)

### Chronic-disease history

| Feature | Type | Description |
|---|---|---|
| `HighBP` | binary | Told they have high blood pressure |
| `HighChol` | binary | Told they have high cholesterol |
| `CholCheck` | binary | Cholesterol check in the past 5 years |
| `Stroke` | binary | Ever told they had a stroke |
| `Diabetes` | ordinal (0/1/2) | 0 = no diabetes, 1 = pre-diabetes or gestational diabetes, 2 = diabetes |

### Behaviours

| Feature | Type | Description |
|---|---|---|
| `Smoker` | binary | Smoked at least 100 cigarettes in lifetime |
| `PhysActivity` | binary | Physical activity in the past 30 days |
| `Fruits` | binary | Eats fruit at least once per day |
| `Veggies` | binary | Eats vegetables at least once per day |
| `HvyAlcoholConsump` | binary | Heavy alcohol consumption indicator |

### Body and health status

| Feature | Type | Description |
|---|---|---|
| `BMI` | continuous | Body Mass Index |
| `GenHlth` | ordinal (1–5) | Self-rated general health: 1 = excellent, 5 = poor |
| `MentHlth` | count (0–30) | Days of poor mental health in the past 30 days |
| `PhysHlth` | count (0–30) | Days of poor physical health in the past 30 days |
| `DiffWalk` | binary | Serious difficulty walking or climbing stairs |

### Healthcare access

| Feature | Type | Description |
|---|---|---|
| `AnyHealthcare` | binary | Any healthcare coverage |
| `NoDocbcCost` | binary | Could not see a doctor in the past 12 months because of cost |

### Demographics

| Feature | Type | Description |
|---|---|---|
| `Sex` | binary | 0 = female, 1 = male |
| `Age` | ordinal (1–13) | Age bucket: 1 = 18–24, ..., 13 = 80+ |
| `Education` | ordinal (1–6) | Education level: 1 = less than elementary, 6 = college graduate |
| `Income` | ordinal (1–8) | Income bracket: 1 = below $10,000, 8 = at least $75,000 |

## Feature handling by method

- **MNN:** z-score standardise all non-binary numeric and ordinal features (`BMI`, `MentHlth`, `PhysHlth`, `Diabetes`, `GenHlth`, `Age`, `Education`, `Income`) using statistics computed from the training fold only.
- **K-means / Hierarchical clustering:** z-score standardise the same non-binary numeric and ordinal features on the exploratory clustering sample, because clustering is not evaluated through supervised train/test folds.
- **Decision Tree:** use raw numeric values; threshold-based splits do not require feature scaling.
- **Naïve Bayes:** treat binary and ordinal features categorically with Laplace smoothing; model `BMI`, `MentHlth`, and `PhysHlth` with class-conditional Gaussian likelihoods.

## Quick load in Python

```python
import pandas as pd

df = pd.read_csv("heart_disease_balanced.csv")
X = df.drop(columns=["target"]).to_numpy()  # shape: (10000, 21)
y = df["target"].to_numpy()                 # shape: (10000,)
```

## Planned evaluation summary

### Balanced primary dataset

- Stratified 5-fold cross-validation
- Confusion matrix
- Accuracy, precision, recall, F1, and false-positive rate
- ROC curves and ROC-AUC
- Permutation feature importance

### Imbalanced secondary dataset

- Separate stratified 5-fold cross-validation
- Majority-class baseline comparison
- Precision, recall, and F1 for the positive class
- ROC-AUC
- Precision-Recall curves and Average Precision (AP)

### Exploratory unsupervised analysis

- K-means on the balanced dataset
- Hierarchical agglomerative clustering with dendrogram visualisation on a stratified subsample for readability

## Reproducibility note

The derived CSV files should be regenerated from `brfss_raw.csv` through `preprocess.py` using the fixed random seed of 42. Before the derived files are committed, the generation pipeline should verify:

- `heart_disease_balanced.csv` contains 10,000 rows with exactly 5,000 positive and 5,000 negative records;
- `heart_disease_clean.csv` contains 15,000 rows with approximately the original positive prevalence;
- the two derived datasets have zero source-row-index overlap.
