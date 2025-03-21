# Green Energy Integration and Investment Analysis for Historic Buildings

## Overview

This project focuses on optimizing the integration of green energy solutions into historic village buildings while preserving their cultural heritage. It also aims to analyze and predict the financial and environmental impacts of green energy investments, helping stakeholders understand the payback period, identify key factors influencing energy efficiency, and simulate the potential benefits of retrofitting measures.

The project is divided into two Notebooks:

**Notebook 1**:
1. Classifying buildings based on solar utilization.
2. Building a recommendation system for green energy solutions.

**Notebook 2**:
3. Predict the payback period for green energy investments.
4. Analyze energy efficiency in historic buildings.
5. Simulate retrofitting scenarios and provide rule-based recommendations.

### Dataset

The dataset used in this project is green_energy_dataset.csv. It contains the following key features:

- Building Characteristics: Year built, floor area, building type, orientation, material, insulation level.
- Energy Metrics: Solar potential, wind potential, geothermal potential, energy demand, installation area.
- Environmental Metrics: Carbon reduction percentage, optimal solar utilization percentage.
- Target Variables: Payback period (in years), solar utilization category (Low, Medium, High).

### Methodology

#### Notebook 1

**1. Data Preprocessing**
- Handle missing values and outliers.
- Convert categorical variables into numerical representations using one-hot encoding.
- Perform feature engineering to create new features (e.g., solar-to-wind ratio, energy demand per m²).
- Standardize numerical features for clustering and regression.

**2. Classify Buildings**

- Use a Random Forest Classifier to categorize buildings into Low, Medium, or High solar utilization.
- Evaluate the model using accuracy, precision, recall, and F1-score.
- Perform hyperparameter tuning to improve performance.

**3. Recommend Solutions**
- Use cosine similarity to recommend buildings with similar characteristics.
- Evaluate the system using Precision@K and Mean Average Precision (MAP).

#### Notebook 2

**4. Predict Payback Period**:
- Train and evaluate regression models (Linear Regression, Random Forest, XGBoost).
- Compare model performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

**5. Building Clustering**:
- Use KMeans clustering to group buildings based on energy demand and carbon reduction potential.
- Visualize clusters using scatter plots.

**6. Retrofitting Simulation**:
- Simulate the impact of retrofitting measures (e.g., improving insulation, increasing solar utilization) on energy demand and carbon reduction.
- Evaluate scenarios with 10%, 20%, and 30% improvements.

### Results

**1. Optimizing Green Energy Integration**

  - The Random Forest Regressor achieved an R² value of 0.7210 and an MSE of 8.2830.
- The most important features are installation area, carbon reduction potential, and solar potential.
- The Random Forest Classifier achieved an accuracy of 72.11%.
- The model performs well for the High and Low categories but struggles with the Medium category.
- The system achieved a Mean Average Precision (MAP) of 0.9444, indicating strong performance on average.
- However, the Precision@3 metric of 0.0000 suggests room for improvement in aligning recommendations with user preferences.

**2. Green Energy Investment Analysis**

**Payback Period Prediction**
- The average payback period is 8.46 years, with a range of 5.01 to 11.99 years.
- Linear Regression performed best, with an MAE of 1.730 and RMSE of 2.005.

**Building Clusters**
Buildings were grouped into three clusters based on energy demand and carbon reduction potential:
- Cluster 0: Moderate energy demand and carbon reduction.
- Cluster 1: Low energy demand, moderate carbon reduction.
- Cluster 2: High energy demand, moderate carbon reduction.

**Retrofitting Simulations**
- Retrofitting measures led to significant energy savings and increased carbon reduction.
- A 30% improvement reduced energy demand by 30% and increased carbon reduction by 30%.

**Rule-Based Recommendations**
- Retrofitting was recommended for buildings with high energy demand, low carbon reduction, or low insulation.

### Future Work

Improve Classification for Medium Utilization: Address class imbalance and explore advanced classification techniques.

### Source

https://www.kaggle.com/datasets/ziya07/green-energy-dataset
