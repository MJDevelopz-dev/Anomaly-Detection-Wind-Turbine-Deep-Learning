# Anomaly Detection in Wind Turbine SCADA Data using LSTM & Bi-LSTM Autoencoders

## Master’s Thesis Project – Anomaly Detection in Wind Turbine Sensor Data using Deep Learning

This project represents part of my **master’s thesis written in Croatian**, titled:  
**“Detekcija anomalija u senzorskim podacima vjetroturbina pomoću dubokog učenja”**  
(*Anomaly Detection in Wind Turbine Sensor Data using Deep Learning*).

The focus of the work is on **real-world multivariate time series SCADA data** from literature, specifically collected from wind turbines in **Germany**, applying modern **feature selection techniques (XGBoost + SHAP)**, and preparing the data for **deep learning-based anomaly detection models**.

In the experimental part of the thesis, **LSTM Autoencoder** and **Bi-LSTM Autoencoder** models were developed and evaluated for **reconstruction-based anomaly detection** in **multivariate time series sensor data** collected from these wind turbines.

---

## Author
* **Mirko Jurišić**  
* **Master’s Degree Programme:** Data Science and Engineering  
* **University of Split – Faculty of Science (PMFST)**  

---

## Thesis Repository
Master’s thesis *(in Croatian)*:  
[https://repozitorij.pmfst.unist.hr/islandora/object/pmfst:2239](https://repozitorij.pmfst.unist.hr/islandora/object/pmfst:2239)

GitHub repository *(in Croatian)*:  
[https://github.com/mirkojurisic/diplomski-rad-projekt](https://github.com/mirkojurisic/diplomski-rad-projekt)

## Dataset

The dataset used in this project originates from the scientific paper  
**[CARE to Compare: A real-world dataset for anomaly detection in wind turbine data](https://zenodo.org/records/15846963)**.

The full dataset consists of **15 event subsets** — **9 normal subsets** and **6 anomaly subsets**.  
For the purpose of this master’s thesis, only the data from **Wind Farm B** was used, specifically the **six subsets containing anomaly events** 


## Files in the Wind Farm B Dataset

### 1. datasets

The folder **datasets** contains the selected anomaly subsets from **Wind Farm B**.  
The CSV files used in this project are:

* **34.csv**  
* **7.csv**  
* **53.csv**  
* **27.csv**  
* **19.csv**  
* **77.csv**

> Note: All experimental preprocessing and evaluation were performed on each of these subsets.

---

### 2. `event_info.csv`

The file **event_info.csv** contains information about recorded events in each subset of Wind Farm **B**.

Each row represents a single event, either **normal** or **anomalous**, and includes:

* **Event description (`event_description`)** – A textual description of the event, often containing information about the cause of the anomaly or turbine failure.

---

### 3. `feature_description.csv`

The file **feature_description.csv** contains:

* A list of all sensors used in the dataset  
* The types of statistics applied  
* Sensor descriptions  
* Corresponding units of measurement  
* Two additional indicators:  

  * **is_angle** – indicates features representing angular values  
  * **is_counter** – indicates sensors functioning as counters

## Structure of the Jupyter Notebook (`Anomaly_Detection_Wind_Turbine_SCADA.ipynb`)

The notebook is organized into sections that follow the complete analysis and experimental workflow — below are the key chapters (as presented in the `.ipynb` file):

1. **Import Libraries**

   * Loading the required libraries (pandas, numpy, matplotlib, xgboost, shap, sklearn, tensorflow/keras)

2. **Dataset**

   * Loading CSV files (from the `Wind Farm B/datasets` folder) and performing initial normalization and checks.

3. **Exploratory Data Analysis (EDA): All Events**

   * Counting normal vs. anomalous events (especially for training and testing sets).  
   * Checking for missing or duplicated records.  
   * Visualizations and summaries per event.

4. **Key EDA Questions**

   * "What is the number of normal and anomalous events?"  
   * "Are there any missing or duplicated data?"  
   * "Which sensor features contribute the most to distinguishing normal and anomalous wind turbine behavior?"

5. **Data Preprocessing, Training, and Testing: All Events**

6. **LSTM Autoencoder**

   * **Without removing low-variance features**  
   * **With removal of low-variance features**

7. **Bi-LSTM Autoencoder**

   * **Without removing low-variance features**  
   * **With removal of low-variance features**

8. **Evaluation and Visualizations**

   * Loss curves, confusion matrices, reconstruction error over time, and histograms.  
   * Detection threshold based on reconstruction error percentiles (**95th percentile** used as the detection threshold in the notebook).

## Feature Selection — Details and Procedure

The feature selection process in the notebook was carried out through two complementary approaches, and the results were compared **before and after the removal of low-variance features**:

1. **Low-Variance Feature Removal**

   * The method `sklearn.feature_selection.VarianceThreshold` was applied with a **threshold of 0.2** to identify and remove features that exhibited very low variability across samples.  
   * All removed features were logged and visually highlighted to illustrate their limited contribution to the model.

2. **XGBoost + SHAP (Feature Importance Ranking)**

   * An **XGBoost binary classification model** was trained on both the filtered and unfiltered feature sets.  
   * The **SHAP (SHapley Additive exPlanations)** framework was used to interpret feature importance, ranking features by their **mean absolute SHAP values**.  
   * The SHAP analysis was conducted **both before and after** applying the variance threshold, and visualized using summary and bar plots.  
   * The **top 5 features** with the **highest mean absolute SHAP values** were selected **both before and after removing low-variance features**, in order to evaluate whether the variance filtering step improved the overall anomaly detection performance.

## Preprocessing

Before training the deep learning models, several preprocessing steps were applied to prepare the SCADA data for time-series anomaly detection:

1. **Data Scaling**

   * All numerical features were standardized using **`sklearn.preprocessing.StandardScaler`** to ensure that each feature had a mean of 0 and a standard deviation of 1.  
   * This normalization helps the neural networks converge faster and prevents features with larger numeric ranges from dominating the training process.

2. **Sliding Window Transformation**

   * The time-series data was transformed into **overlapping sliding windows** to preserve temporal dependencies between consecutive measurements.  
   * A **timestep of 6** was used, corresponding to **1 hour of data**, since each row in the SCADA dataset represents a **10-minute interval**.  
   * This transformation allowed the LSTM and Bi-LSTM Autoencoder models to learn from short-term temporal patterns that may indicate early signs of anomalies.

## Model Hyperparameters

Below are the hyperparameter configurations used for the **LSTM Autoencoder** and **Bi-LSTM Autoencoder** models in the anomaly detection experiments.

---

### **Table 1. Hyperparameters for the LSTM Autoencoder**

| Hyperparameter | Value |
|----------------|--------|
| Timesteps | 6 |
| Number of LSTM layers (Encoder/Decoder) | 1 |
| Number of neurons in LSTM layer (Encoder/Decoder) | 10 |
| Activation function | tanh |
| Dropout | 0.2 |
| Optimizer | Adam (learning rate = 0.0001) |
| Loss function | Mean Absolute Error (MAE) |
| Number of epochs | 30 |
| Batch size | 256 |
| Validation split | 0.1 |

---

### **Table 2. Hyperparameters for the Bi-LSTM Autoencoder**

| Hyperparameter | Value |
|----------------|--------|
| Timesteps | 6 |
| Number of Bi-LSTM layers (Encoder/Decoder) | 1 |
| Number of neurons in Bi-LSTM layer (Encoder/Decoder) | 10 |
| Activation function | tanh |
| Dropout | 0.2 |
| Optimizer | Adam (learning rate = 0.0001) |
| Loss function | Mean Absolute Error (MAE) |
| Number of epochs | 15 |
| Batch size | 256 |
| Validation split | 0.1 |

## Results — LSTM Autoencoder

### Table 3. LSTM Autoencoder Results (without removing low-variance features)

| Event Subset | Accuracy | Precision | Recall | F1 Score | FAR |
|--------------|---------|-----------|--------|----------|-----|
| Event 34     | 93.75%  | 36.36%    | 100%   | 53.33%   | 6.48% |
| Event 7      | 94.88%  | 60.68%    | 98.56% | 75.11%   | 5.44% |
| Event 53     | 95.37%  | 54.62%    | 100%   | 70.65%   | 4.90% |
| Event 27     | 97.12%  | 81.32%    | 100%   | 89.70%   | 3.29% |
| Event 19     | 94.15%  | 18.28%    | 100%   | 30.91%   | 5.93% |
| Event 77     | 95.55%  | 70.80%    | 100%   | 82.90%   | 4.99% |
| **Average**  | **95.14%** | **53.68%** | **99.76%** | **67.10%** | **5.17%** |

### Table 4. LSTM Autoencoder Results (with low-variance feature removal)

| Event Subset | Accuracy | Precision | Recall | F1 Score | FAR |
|--------------|---------|-----------|--------|----------|-----|
| Event 34     | 94.97%  | 41.50%    | 100%   | 58.66%   | 5.22% |
| Event 7      | 97.62%  | 76.70%    | 100%   | 86.81%   | 2.59% |
| Event 53     | 94.53%  | 50.45%    | 100%   | 67.06%   | 5.80% |
| Event 27     | 98.36%  | 88.42%    | 100%   | 93.86%   | 1.88% |
| Event 19     | 96.15%  | 25.39%    | 100%   | 40.50%   | 3.90% |
| Event 77     | 95.67%  | 71.36%    | 100%   | 83.28%   | 4.85% |
| **Average**  | **96.22%** | **58.97%** | **100%** | **71.69%** | **4.04%** |

## Results — Bi-LSTM Autoencoder

### Table 5. Bi-LSTM Autoencoder Results (without removing low-variance features)

| Event Subset | Accuracy | Precision | Recall | F1 Score | FAR |
|--------------|---------|-----------|--------|----------|-----|
| Event 34     | 94.45%  | 39.13%    | 100%   | 56.25%   | 5.76% |
| Event 7      | 95.05%  | 61.49%    | 98.56% | 75.74%   | 5.25% |
| Event 53     | 95.27%  | 54.09%    | 100%   | 70.21%   | 5.01% |
| Event 27     | 97.63%  | 84.06%    | 100%   | 91.34%   | 2.72% |
| Event 19     | 94.34%  | 18.77%    | 100%   | 31.61%   | 5.74% |
| Event 77     | 95.79%  | 71.92%    | 100%   | 83.67%   | 4.72% |
| **Average**  | **95.42%** | **54.91%** | **99.76%** | **68.14%** | **4.87%** |

### Table 6. Bi-LSTM Autoencoder Results (with low-variance feature removal)

| Event Subset | Accuracy | Precision | Recall | F1 Score | FAR |
|--------------|---------|-----------|--------|----------|-----|
| Event 34     | 94.82%  | 40.79%    | 100%   | 57.95%   | 5.37% |
| Event 7      | 96.64%  | 70.02%    | 100%   | 82.36%   | 3.64% |
| Event 53     | 95.17%  | 53.58%    | 100%   | 69.77%   | 5.11% |
| Event 27     | 98.36%  | 88.42%    | 100%   | 93.86%   | 1.88% |
| Event 19     | 96.98%  | 30.25%    | 100%   | 46.45%   | 3.06% |
| Event 77     | 95.78%  | 71.87%    | 100%   | 83.63%   | 4.73% |
| **Average**  | **96.29%** | **59.16%** | **100%** | **72.34%** | **3.96%** |
