# 👨‍💼 Employee Attrition Prediction App

This is an end-to-end machine learning project aimed at predicting the likelihood of an employee leaving (attrition) a company. The project is implemented as an interactive web application built with Streamlit, allowing HR teams or management to input employee data and receive a real-time risk score.

---

## 🎯 Problem Statement

The high costs associated with recruitment and talent loss due to employee attrition are critical issues for many companies. By understanding the key factors that influence an employee's decision to leave, organizations can take proactive steps to improve retention and keep their top talent.

**Project Goals:**
* Analyze historical employee data to find patterns indicative of attrition risk.
* Build an accurate machine learning model to predict the probability of attrition.
* Present the model in an easy-to-use application for non-technical stakeholders (like the HR team).

---

## 🚀 Key Features

* **Interactive Input Form:** Users can input various employee data points such as age, monthly income, overtime status, and more.
* **Real-time Prediction:** The application instantly provides an attrition risk score as a percentage after the data is submitted.
* **Result Interpretation:** Offers simple recommendations based on the calculated risk score.

---

## 🛠️ Tech Stack & Libraries

This project was built using the Python ecosystem with the following core libraries:

* **Data Analysis & Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `imbalanced-learn`
* **Frontend Web App:** `streamlit`
* **Data Visualization (Evaluation):** `matplotlib`, `seaborn`
* **Model Persistence:** `joblib`
* **Reading Excel Files:** `openpyxl`

---

## 📂 Project Structure

The folder and file structure is organized to be modular and easy to understand, separating each stage of the machine learning workflow.

.
├── 📂 data/
│   ├── X_train_resampled.npy
│   ├── y_train_resampled.npy
│   ├── X_test_processed.npy
│   └── y_test.npy
├── 📄 HR_Employee_Attrition.xlsx
├── 📄 app.py                     # Main Streamlit application script
├── 📄 attrition_model.joblib     # Trained model
├── 📄 evaluate_model.py          # Script for model evaluation
├── 📄 label_encoder.joblib       # LabelEncoder object
├── 📄 preprocessing.py           # Script for data preprocessing
├── 📄 preprocessor.joblib        # ColumnTransformer object
├── 📄 requirements.txt           # List of required libraries
└── 📄 README.md                  # This documentation

---

## ⚙️ Methodology & Project Workflow

This project follows a standard machine learning workflow consisting of several stages:

1.  **Exploratory Data Analysis (EDA):** Analyzed and visualized the data to understand distributions, relationships between features, and identify key variables most influential to attrition. It was found that the data is imbalanced, with only ~16% of employees having attrited.
2.  **Data Preprocessing (`preprocessing.py`):**
    * **Cleaning:** Dropped irrelevant columns.
    * **Encoding:** Used `LabelEncoder` for the target variable and `OneHotEncoder` for categorical features.
    * **Handling Imbalanced Data:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) only on the training data to balance the classes.
    * **Scaling:** Used `StandardScaler` on numerical features.
3.  **Model Training (`train_model.py`):**
    * The model of choice was the **Random Forest Classifier**, a powerful and stable ensemble model for classification problems.
    * The model was trained on the processed and balanced data.
4.  **Model Evaluation (`evaluate_model.py`):**
    * Evaluated the model's performance on the unseen test data.
    * Focused on metrics like **Precision**, **Recall**, and **AUC-ROC** due to the imbalanced nature of the data.
5.  **Frontend Development (`app.py`):**
    * Built an interactive user interface with **Streamlit**.
    * The application loads all saved artifacts (`preprocessor`, `model`, `encoder`) to make predictions on new data entered by the user.

---

## 📈 Model Results

The evaluation results showed that the model has good predictive capability, but with a specific characteristic:

* **Confusion Matrix:** The model is very accurate at identifying employees who will **NOT** attrite (high True Negatives).
* **Recall (Attrition Yes):** The model is more conservative and has a lower recall for the attrition class (around 25.5%). This means it might not "catch" all at-risk employees.
* **Application Output:** Due to the low recall, the application was designed to display a **probability risk score** rather than just a binary "Yes/No" prediction. This is more useful for users, as they can follow up even on employees with a moderate risk score.

---

## 🔧 Local Setup & Installation

To run this application on your local machine, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/firdauzan1/attrition-prediction-analysis.git](https://github.com/firdauzan1/attrition-prediction-analysis.git)
    cd attrition-prediction-analysis
    ```

2.  **Create and Activate a Virtual Environment** (Highly recommended)
    ```bash
    # Create venv
    python -m venv venv

    # Activate venv (Windows)
    .\venv\Scripts\activate

    # Activate venv (macOS/Linux)
    source venv/bin/activate
    ```

3.  **Install Required Libraries**
    Ensure you are in the project's root directory, then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit Application**
    After all installations are complete, run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
    The application will automatically open in your web browser.

---
