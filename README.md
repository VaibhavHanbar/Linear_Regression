# Linear_Regression
Insurance Charges Prediction
Machine Learning Pipeline with PCA and Linear Regression

1. Introduction
This document outlines the development of a predictive model for estimating insurance charges using a structured machine learning pipeline. The model achieves 78% accuracy by leveraging Principal Component Analysis (PCA) for dimensionality reduction and Linear Regression for predictions.

3. Dataset Description
The dataset includes the following features:

Feature 	 Type	        Description
age	      Numerical	   Age of the policyholder
bmi	      Numerical	   Body Mass Index
children  Numerical	   Number of dependents
sex	      Categorical	 Gender (male/female)
smoker	  Categorical	 Smoking status (yes/no)
region 	  Categorical	 Geographic region
charges	  Numerical	   Insurance charges (target)

3. Methodology
3.1 Exploratory Data Analysis (EDA)
Missing Values: Verified no null entries using df.isnull().sum().
Distributions: Visualized using sns.histplot() and sns.boxplot().
Correlations: Analyzed with sns.heatmap().

3.2 Data Preprocessing
Numerical Features: Standardized using StandardScaler().
Categorical Features: Encoded via OneHotEncoder(drop='first').

4. Results
Metric   	             Value
R² Score	             0.78
PCA Explained Variance	95%

Key Insights:
The first two principal components explain 85% of variance (PC1: 60%, PC2: 25%).
Feature importance analysis revealed smoker and age as top predictors.

5. Recommendations for Improvement
Model Selection: Test RandomForestRegressor or XGBoost for non-linear relationships.
Hyperparameter Tuning: Optimize PCA components and regression parameters via GridSearchCV.
Feature Engineering: Explore interaction terms (e.g., age × bmi).

7. Conclusion
This pipeline provides a robust framework for predicting insurance charges with 78% accuracy. Future work should focus on enhancing performance through advanced modeling techniques and feature optimization.
