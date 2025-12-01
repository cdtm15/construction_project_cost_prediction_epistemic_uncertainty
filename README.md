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
