# Cracking the Code: Investigating Advanced Intrusion Detection Frameworks for SCADA Security 

## Overview
This repository contains code and analysis for implementing advanced intrusion detection techniques in SCADA (Supervisory Control and Data Acquisition) systems. SCADA systems are critical for managing industrial operations, but their increasing connectivity to the internet exposes them to cyber threats such as buffer overflows, SQL injection, and cross-site scripting. This project investigates how advanced machine learning and statistical methods can identify and mitigate unauthorized access or anomalies in SCADA systems in real-time.

The work focuses on the **Gas Pipeline Dataset**, utilizing state-of-the-art techniques to improve the resilience and security of SCADA environments. The dataset comprises 274,628 instances across multiple classes and covers 7 types of cyberattacks, making it an ideal choice for evaluating intrusion detection strategies.

---

## Project Objectives
1. **Implement Advanced Detection Techniques**: Develop and compare machine learning models, such as Random Forest and Convolutional Neural Networks, for detecting intrusions in SCADA systems.
2. **Enhance SCADA Security**: Analyze key vulnerabilities, particularly in the Modbus protocol, and evaluate strategies to enhance system resilience.
3. **Performance Benchmarking**: Utilize metrics such as accuracy, precision, recall, and F1 score to identify optimal intrusion detection approaches.

---
## Dataset
- **Source**: [Gas Pipeline Dataset](http://www.ece.uah.edu/~thm0009/icsdatasets/IanArffDataset.arff)
- **Features**: 17 columns, including command payload features, network characteristics, and response payloads.
- **Labels**: Binary, categorized, and specific results for various types of cyberattacks.
- **Notable Attributes**: Timestamp, source/destination IP, protocol, packet size, and 11 command payload features.

---

## Methodology
1. **Data Preprocessing**:
   - Cleaning and handling missing values.
   - Normalizing data for consistency.
   - Splitting the dataset into training and testing sets.
2. **Model Development**:
   - **Baseline Models**: Random Forest and Decision Trees KNN, Random Forest, Naive Bayes, MLP for initial analysis. 
   - **Advanced Models**: Stacked neural networks using Convolutional Neural Networks (CNNs) with ReLU activation and batch normalization. LSTM and GRU networks for sequence modeling.
3. **Evaluation**:
   - **Metrics**: Accuracy, precision, recall, F1 score, ROC-AUC and confusion matrices.
   - Comparative analysis of traditional and deep learning models.

---

## Code Files
### 1. `Baseline_Model_Comparison.ipynb`
   - Implements baseline models such as Random Forest and Decision Trees.
   - Provides exploratory data analysis and pre-processing steps.

### 2. `Stacked_NN.ipynb`
   - Implements a stacked neural network with advanced configurations like batch normalization.
   - Evaluates performance on the test dataset and provides insights on classification accuracy.

### 3. `ICS-IDS.ipynb`
   - The code cleans, standardizes, and balances the dataset using ENN for robust model training.
   - Trains and evaluates ML (KNN, Random Forest) and DL (MLP, LSTM, GRU) models with performance visualizations.

---

## Results
- **Baseline Models**: Achieved ~85% accuracy with Random Forest and Decision Trees.
- **Advanced Models**: Stacked Neural Networks improved detection rates, achieving up to 93% accuracy.
- **ICS-IDS**: Combining multiple techniques like feature selection and deep learning models yielded the most robust results. GRU achieved 85.4%, LSTM achieved 84.9%, Random Forest achieving approximately 84%.

---

## Technologies and Tools
- **Languages**: Python
- **Libraries**: Scikit-learn, TensorFlow, Pandas, NumPy
- **Tools**: Jupyter Notebook, SciPy, Matplotlib
- **Dataset Format**: ARFF and CSV

---

## Future Work
- Extend analysis to include real-time deployment of intrusion detection systems.
- Evaluate scalability for larger SCADA systems with live data streams.
- Explore additional cybersecurity frameworks like hybrid anomaly detection methods.

---

## References
1. [Intrusion Detection and Identification System Design and Performance Evaluation for Industrial SCADA Networks]([https://example.com/paper1](https://arxiv.org/abs/2012.09707))  
2. [A Stacked Deep Learning Approach to Cyber-Attacks Detection in Industrial Systems]([https://example.com/paper2](https://link.springer.com/article/10.1007/s10586-021-03426-w))  
3. [ICS-IDS: Application of Big Data Analysis in AI-Based Intrusion Detection Systems for SCADA Networks]([https://example.com/paper3](https://link.springer.com/article/10.1007/s11227-023-05764-5#Sec7))  

For further details, please refer to the project proposal and accompanying documentation.
