# Mastering AI 03 - Introduction to Machine Learning

Welcome to the **Mastering AI 03 - Introduction to Machine Learning** repository! This document provides a comprehensive roadmap to guide your learning journey through the foundational concepts and practical applications of Machine Learning (ML). Each section is designed to build upon the previous one, ensuring a thorough understanding of ML principles and techniques.

## Table of Contents

1. [Introduction to Machine Learning](#1-introduction-to-machine-learning)
2. [Basic Supervised Learning](#2-basic-supervised-learning)
3. [Basic Unsupervised Learning](#3-basic-unsupervised-learning)
4. [Model Evaluation and Selection](#4-model-evaluation-and-selection)
5. [Feature Engineering](#5-feature-engineering)
6. [Basic Ensemble Methods](#6-basic-ensemble-methods)
7. [Practical Considerations](#7-practical-considerations)
8. [Tools and Libraries](#8-tools-and-libraries)

---

## 1. Introduction to Machine Learning

### 1.1 Definition of Machine Learning
- What is ML?
- Historical context and evolution.

### 1.2 Types of Learning
- **Supervised Learning**: Algorithms learn from labeled data.
- **Unsupervised Learning**: Algorithms learn from unlabeled data.
- **Semi-Supervised Learning**: Combines labeled and unlabeled data.

### 1.3 Use Cases
- **Spam Detection**: Identifying unwanted emails.
- **Recommendation Systems**: Suggesting products or content.
- **Predictive Analytics**: Forecasting future trends.

#### Questions
1. What are the key differences between supervised and unsupervised learning in terms of data requirements and model objectives?
2. How does semi-supervised learning leverage both labeled and unlabeled data to improve model performance?
3. What are some real-world applications where supervised learning is preferred over unsupervised learning?
4. In what scenarios might unsupervised learning provide more insights than supervised learning?
5. How can the use cases of machine learning in spam detection and recommendation systems be compared in terms of model complexity and data requirements?

---

## 2. Basic Supervised Learning

### 2.1 Linear Regression
- **Simple Linear Regression**: Model with one feature.
- **Multiple Linear Regression**: Model with multiple features.
- **Loss Functions**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE).
- **Optimization**: Gradient Descent, Learning Rate.

### 2.2 Logistic Regression
- **Binary Classification**: Output class probabilities.
- **Sigmoid Function**: Transformation to [0, 1] range.
- **Cost Function**: Binary Cross-Entropy.
- **Regularization**: L1 (Lasso), L2 (Ridge).

### 2.3 Decision Trees
- **Tree Structure**: Nodes, branches, leaves.
- **Splitting Criteria**: Gini Impurity, Entropy, Information Gain.
- **Tree Pruning**: Reducing complexity to prevent overfitting.

### 2.4 Support Vector Machines (SVM)
- **Concept of Margin**: Maximizing the distance between classes.
- **Support Vectors**: Critical data points for defining the margin.
- **Kernels**: Linear, Polynomial, Radial Basis Function (RBF).

### 2.5 k-Nearest Neighbors (k-NN)
- **Distance Metrics**: Euclidean, Manhattan.
- **Choosing k**: Balancing bias and variance.
- **Classification vs. Regression**: Handling different types of problems.

### 2.6 Naive Bayes
- **Bayes Theorem**: Probability calculations.
- **Conditional Independence**: Assumption in Naive Bayes.
- **Types**: Gaussian Naive Bayes, Multinomial Naive Bayes.

#### Questions
1. How do the assumptions of linear regression compare with those of logistic regression in terms of model flexibility and application?
2. What are the trade-offs between using decision trees and support vector machines for classification tasks?
3. How does the choice of distance metric in k-NN impact the model's performance and computational complexity?
4. In what situations might Naive Bayes outperform more complex models like decision trees?
5. How does regularization in logistic regression affect model performance compared to pruning in decision trees?

---

## 3. Basic Unsupervised Learning

### 3.1 Clustering
- **k-Means Clustering**: Centroid-based clustering, choosing k.
- **Hierarchical Clustering**: Agglomerative and Divisive methods.
- **Density-Based Clustering**: DBSCAN, identifying clusters of varying shapes.

### 3.2 Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Reducing dimensions while preserving variance.
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: Visualizing high-dimensional data.

### 3.3 Anomaly Detection
- **Isolation Forest**: Detecting anomalies by isolating observations.
- **One-Class SVM**: Classification for anomaly detection.

#### Questions
1. How does k-Means clustering compare to hierarchical clustering in terms of scalability and interpretability?
2. What are the strengths and weaknesses of PCA versus t-SNE for dimensionality reduction?
3. How does DBSCAN handle clusters of varying shapes compared to k-Means?
4. What are the main differences between Isolation Forest and One-Class SVM for anomaly detection?
5. In what scenarios might dimensionality reduction techniques improve the performance of clustering algorithms?

---

## 4. Model Evaluation and Selection

### 4.1 Metrics
- **Classification Metrics**:
  - **Accuracy**: Overall correctness.
  - **Precision**: True positives / (True positives + False positives).
  - **Recall**: True positives / (True positives + False negatives).
  - **F1 Score**: Harmonic mean of precision and recall.
  - **ROC Curve**: Receiver Operating Characteristic curve.
  - **AUC-ROC**: Area Under the Curve - ROC.
- **Regression Metrics**:
  - **Mean Absolute Error (MAE)**: Average of absolute errors.
  - **Mean Squared Error (MSE)**: Average of squared errors.
  - **R² Score**: Proportion of variance explained by the model.

### 4.2 Validation Techniques
- **Cross-Validation**:
  - **k-Fold Cross-Validation**: Splitting data into k subsets.
  - **Leave-One-Out Cross-Validation**: Each sample is used once as a test set.
- **Train-Test Split**: Dividing data into training and testing sets.

### 4.3 Bias-Variance Tradeoff
- **Underfitting**: Model is too simple.
- **Overfitting**: Model is too complex.
- **Balancing Bias and Variance**: Choosing appropriate model complexity.

### 4.4 Hyperparameter Tuning
- **Grid Search**: Exhaustive search over specified parameter values.
- **Random Search**: Random sampling of hyperparameters.
- **Bayesian Optimization**: Probabilistic model to optimize hyperparameters.

#### Questions
1. How do accuracy and F1 score differ in evaluating model performance, especially in imbalanced datasets?
2. What are the advantages and disadvantages of k-Fold Cross-Validation compared to Leave-One-Out Cross-Validation?
3. How can the choice of validation technique impact the reliability of model performance estimates?
4. What are the practical implications of the bias-variance tradeoff on model selection and tuning?
5. How do Grid Search and Random Search compare in terms of computational efficiency and effectiveness in hyperparameter tuning?

---

## 5. Feature Engineering

### 5.1 Feature Selection
- **Filter Methods**: Statistical techniques (e.g., correlation, Chi-Square test).
- **Wrapper Methods**: Feature selection using a predictive model (e.g., Recursive Feature Elimination).
- **Embedded Methods**: Feature selection integrated with model training (e.g., Lasso).

### 5.2 Feature Extraction
- **Polynomial Features**: Adding polynomial terms.
- **Interaction Terms**: Creating features from interactions between variables.

### 5.3 Scaling and Normalization
- **Standardization**: Transforming features to zero mean and unit variance.
- **Normalization**: Scaling features to a specific range (e.g., [0, 1]).
- **Robust Scaling**: Scaling features using statistics that are robust to outliers.

#### Questions
1. How do filter methods for feature selection compare with wrapper methods in terms of computational efficiency and accuracy?
2. What are the trade-offs between using polynomial features and interaction terms for feature extraction?
3. How does standardization differ from normalization in terms of impact on model performance?
4. When would robust scaling be preferred over standardization or normalization?
5. How do feature selection and feature extraction techniques affect the interpretability and complexity of machine learning models?

---

## 6. Basic Ensemble Methods

### 6.1 Bagging
- **Bootstrap Aggregating**: Training multiple models on different subsets.
- **Random Forest**: Ensemble of decision trees with random feature selection.

### 6.2 Boosting
- **AdaBoost**: Combining weak learners to create a strong learner.
- **Gradient Boosting**: Iteratively improving the model by correcting errors.
- **XGBoost**: Optimized gradient boosting with regularization.

### 6.3 Stacking
- **Stacked Generalization**: Combining predictions from multiple models using a meta-learner.

#### Questions
1. How does the concept

 of bagging differ from boosting in terms of model training and error reduction?
2. What are the strengths and weaknesses of Random Forest compared to Gradient Boosting?
3. How does XGBoost enhance the performance of gradient boosting methods?
4. In what scenarios might stacking models be more effective than using individual ensemble methods?
5. How does the choice of ensemble method impact model interpretability and computational requirements?

---

## 7. Practical Considerations

### 7.1 Data Preprocessing
- **Handling Missing Values**: Imputation methods (mean, median, mode).
- **Encoding Categorical Variables**: One-hot encoding, Label encoding.
- **Data Augmentation**: Techniques to expand training data (e.g., rotations, flips).

### 7.2 Model Deployment
- **Saving Models**: Techniques for persisting models (e.g., joblib, pickle).
- **Simple Deployment Methods**: Serving models using Flask or FastAPI.

### 7.3 Ethics and Bias
- **Identifying Bias**: Techniques for detecting biases in data and models.
- **Mitigating Bias**: Methods to reduce bias (e.g., balanced datasets).
- **Ethical Implications**: Considerations for responsible AI use.

#### Questions
1. What are the different methods for handling missing values, and how do they impact model performance?
2. How do one-hot encoding and label encoding compare in terms of handling categorical variables?
3. What are the trade-offs between using simple data augmentation techniques versus more complex methods?
4. How does saving and deploying models differ in terms of format and usability?
5. What are the challenges in identifying and mitigating bias in machine learning models, and how can ethical implications be addressed?

---

## 8. Tools and Libraries

### 8.1 Scikit-Learn
- **Core Functions**: Model training and evaluation.
- **Pipeline**: Creating end-to-end ML pipelines.

### 8.2 Pandas
- **Data Manipulation**: DataFrames, handling missing values.
- **Data Cleaning**: Filtering, merging, and transforming data.

### 8.3 Matplotlib and Seaborn
- **Data Visualization**: Creating plots and graphs for data exploration.

#### Questions
1. How does Scikit-Learn's pipeline feature streamline the process of model training and evaluation?
2. What are the advantages of using Pandas for data manipulation compared to other libraries like NumPy?
3. How do Matplotlib and Seaborn compare in terms of ease of use and visualization capabilities?
4. What are the best practices for integrating data cleaning and manipulation with model training?
5. How can visualization libraries help in interpreting the results of machine learning models and guiding further analysis?

---
