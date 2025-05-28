# Loan Approval Prediction Using Machine Learning

This project explores the application of various machine learning algorithms to predict loan approval outcomes based on applicant data. It aims to streamline the decision-making process for financial institutions by using historical loan data and powerful ML models to assess credit risk efficiently and fairly.

## 🔍 Project Overview

Traditional loan approval processes can be time-consuming and prone to biases. Our goal was to build a robust predictive model using historical data to assist financial institutions in making informed and equitable lending decisions. 

We evaluated multiple machine learning classifiers including:
- Logistic Regression
- Support Vector Classifier (SVC)
- Decision Tree Classifier
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)

Among these, the **Random Forest Classifier** achieved the highest accuracy and was selected as the final model.

## 📊 Dataset

- **Source:** [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Sample Size:** 615 records
- **Features:**
  - Loan ID
  - Gender
  - Marital Status
  - Dependents
  - Education
  - Employment
  - Applicant Income
  - Loan Amount
  - Credit History
  - Property Area

## ⚙️ Tech Stack

- **Language:** Python
- **Platform:** Google Colab
- **Libraries Used:**
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

## 🧪 Methodology

### Data Preprocessing
- Handling missing values and outliers
- Encoding categorical variables
- Feature scaling
- Splitting dataset into training, validation, and testing sets

### Model Training & Evaluation
- Hyperparameter tuning
- Evaluation using metrics: Accuracy, Precision, Recall, F1-score
- Selection of best-performing model (Random Forest)

## ✅ Results

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | 81%      |
| SVC                 | 82%      |
| Decision Tree       | 94%      |
| **Random Forest**   | **95%**  |
| Gradient Boosting   | 88%      |
| KNN                 | 78%      |

The **Random Forest Classifier** emerged as the top performer due to its robustness and accuracy.

## 📦 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/loan-approval-prediction.git
   cd loan-approval-prediction
   ```

2. Run through each cell of `Loan_Approval_Prediction.ipynb`.

## 📜 License

This project is for educational purposes only. Feel free to fork or cite with proper attribution.
