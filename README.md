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

### Visualizations

#### SHAP Analysis

![SHAP Analysis](output_visualizations/Captura%20de%20pantalla%202025-12-01%20a%20la(s)%2012.06.39%E2%80%AFp.m..png)

#### Construction cost prediction with epistemic uncertainty
![Cost prediction analysis](output_visualizations/Captura%20de%20pantalla%202025-12-01%20a%20la(s)%2012.07.01%E2%80%AFp.m..png)


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