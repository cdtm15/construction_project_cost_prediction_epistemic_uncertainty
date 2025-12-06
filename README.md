# A Data Intelligence Framework for Construction Cost Estimation Under Market and Epistemic Uncertainty

The code implements a framework integrating data preprocessing, clustering, feature importance, and regression models to predict construction costs under fluctuating market conditions. The framework also addresses epistemic uncertainty by determining the minimum number of features required for stable predictions.

### Overview
This project provides:
- A CRISP-DM inspired pipeline for cost-estimation modeling.
- Automated preprocessing, feature engineering, encoding and cleaning.
- Hierarchical clustering to detect latent project typologies (Type 0 / Type 1).
- SHAP-based explainability for project-specific and market variables.
- Incremental sensitivity analysis to assess how prediction accuracy varies as more market variables are included.

Training and evaluation of:
- Decision Trees
- Random Forest
- Logistic Regression
- KNN
- SVM
- Neural Networks (TensorFlow)

### Repository structure

```bash
.
├── main.py                       # Main execution script
├── file_data_reception_db2.py    # Data loading and initial parsing
├── file_data_understanding_db2.py# Descriptive stats and correlations
├── file_data_preparation_db2.py  # IQR filtering, encoding, clustering
├── file_modeling_db2.py          # Classification models (optional)
├── file_modeling_regresion_db2.py# Regression models and incremental analysis
├── input_data/                   # Construction projects with market variables dataset
└── output_visualizations/        # SHAP plots, scatter plots
```

### Results

#### SHAP and Model Performance Incremental Sensitivity Analysis for Large Scale Projects

The figure reveals which economic indicators (land price index, gold price and construction cost of buildings by private sector at the time of completion of construction) produce meaningful improvements in predictive accuracy, and identifies the exact point where additional features no longer add value.

This has direct business relevance: companies can focus on the minimal set of economic variables that actually impact project uncertainty and financial performance.

![SHAP Analysis and Models Performance](output_visualizations/Captura%20de%20pantalla%202025-12-06%20a%20la(s)%204.43.57%E2%80%AFp.m..png)


### Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/cdtm15/construction_project_cost_prediction_epistemic_uncertainty.git
cd construction-cost-uncertainty
```

### How to run

```bash
python main_db2.py
```
The script will:
- Load and preprocess data
- Generate SHAP analysis
- Train ANN, SVM, RF for each cluster

### Contributions
Contributions and extensions are welcome in the following areas:
- Bayesian / bootstrap uncertainty quantification
- Integration into BIM platforms
- Extensions to other construction datasets
- Conformal or Quantile Conformal regression models
- Feature selection enhancements beyond SHAP

# License
This project is licensed under the MIT License – see the LICENSE file for details.