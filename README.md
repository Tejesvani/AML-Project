# 🧠 Suspicious Transaction Detection using CNN-LSTM Hybrid Networks

A deep learning approach for identifying suspicious financial transactions in anti money laundering (AML) systems. This project compares multiple architectures — Dense Neural Networks, LSTMs, and CNN-LSTM hybrids — to detect subtle, temporally-sensitive anomalies in realistic synthetic transaction data.

---

## 🎯 Objective

To design, develop, and evaluate a robust deep learning framework capable of detecting suspicious financial transactions, improving upon traditional rule-based and classical ML systems by leveraging both temporal and behavioral cues in transaction data.

---

## 🧾 Abstract

This project presents a hybrid deep learning model that combines Convolutional Neural Networks (CNN) with Long Short-Term Memory (LSTM) networks to detect suspicious transactions within the SAML-D synthetic AML dataset. The hybrid model exploits CNNs for detecting localized anomalies and LSTMs for learning long-term behavioral patterns in transaction sequences. Through rigorous preprocessing, feature engineering, and resampling techniques, the model achieves improved precision, recall, and PR AUC scores, making it suitable for real-world financial monitoring scenarios.

---

## 📦 Dataset

**SAML-D (Synthetic AML Dataset)**  
- ~9.5 million transaction records  
- 12 core features, including time, amount, currency, and account identifiers  
- 28 transaction typologies: 11 normal, 17 suspicious  
- 0.1039% of transactions labeled suspicious, mimicking real-world imbalance  
- Includes behavioral features and network-based transaction flow  

**[Dataset](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)** - This dataset was designed with guidance from AML experts to simulate realistic laundering scenarios, preserving privacy while offering structural and statistical authenticity.

---

## 🔬 Approach

1. **Data Cleaning**  
   - Merged timestamp fields  
   - Currency normalization to USD  
   - Removed leakage-prone features  
   - Handled categorical inconsistencies and missing values  
   - Typecast numeric fields for memory efficiency  

2. **Feature Engineering**  
   - Temporal features: Hour, Day, Weekday (sine/cosine transformed)  
   - Behavioral profiling: Aggregated stats per account (mean, std, count, etc.)  
   - Laundering ratio per account  

3. **Handling Class Imbalance**  
   - SMOTE oversampling  
   - Random undersampling  
   - Threshold optimization to fine-tune precision/recall  

4. **Modeling Strategy**  
   - Implemented and evaluated three progressively complex deep learning architectures  
   - Used early stopping with PR AUC as the primary metric  
   - Evaluation based on ROC AUC, PR AUC, precision, recall, and F1-score

---

## 🧰 Key Technologies

- Python (Pandas, NumPy)
- TensorFlow / Keras
- Matplotlib & Seaborn
- SMOTE (Imbalanced-learn)
- Scikit-learn (for metrics and preprocessing)
- Jupyter Notebooks

---

## 🏗️ Models and Rationale

| Model               | Purpose & Architecture                                                                                      | Strengths                                                 | Weaknesses                                   |
|--------------------|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|----------------------------------------------|
| **Dense NN**        | Baseline model using static features (3 layers, dropout, L2 regularization)                                 | Simple and fast, good initial benchmark                    | Lacks temporal modeling; lower precision     |
| **LSTM**            | Learns temporal patterns from grouped sender sequences (2 LSTM layers, dropout, batch norm)                  | Captures time-dependency in transactions                   | Misses short-term/localized anomalies        |
| **Hybrid CNN-LSTM** | Parallel branches: Conv1D layers capture localized trends, LSTM captures sequence dependencies; merged layer | Best of both worlds—detects both local and global behaviors | More complex, higher training cost           |

---

## 📊 Performance Summary

| Metric     | Dense NN   | LSTM       | CNN-LSTM   |
|------------|------------|------------|------------|
| Precision  | Moderate   | High       | **Highest**|
| Recall     | High       | Higher     | **Highest**|
| PR AUC     | Good       | Better     | **Best**   |
| ROC AUC    | Fair       | Good       | **Excellent**|

Threshold tuning on the hybrid model produced the best F1-score, reducing false positives and false negatives.

---

## 🔍 Key Findings

- The CNN-LSTM hybrid architecture significantly outperformed simpler models.
- Capturing both spatial (transaction-level) and temporal (behavioral) dynamics is essential in fraud detection.
- Feature engineering (e.g., cyclic time features and behavioral stats) played a crucial role in boosting model performance.
- Class imbalance mitigation and careful threshold tuning greatly influenced final model accuracy and stability.

---

## 🚀 Future Enhancements

- Incorporate attention mechanisms to dynamically weigh transaction sequences.
- Apply model interpretability tools like SHAP or LIME to enhance transparency for compliance teams.
- Test generalization on multilingual and multi-currency datasets.
- Integrate with real-time streaming systems for deployment in live AML monitoring pipelines.
- Explore federated learning or privacy-preserving methods for secure financial applications.
