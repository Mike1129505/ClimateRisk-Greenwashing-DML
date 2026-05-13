# ClimateRisk-Greenwashing-DML
# Climate Risk and Corporate Greenwashing: A DML Framework

This repository contains the Python implementation for the research paper: 
*"The Impact of Climate Risk on Corporate Greenwashing: Evidence from Debiased Machine Learning."*

## 🛠 Project Overview
This project investigates how physical and transition climate risks influence strategic environmental disclosure (greenwashing) using a **Debiased Machine Learning (DML)** framework. The code addresses high-dimensional confounding and potential non-linear relationships between firm characteristics and greenwashing behaviors.

##  Key Features in the Code
The core analysis script (`tune_and_check.py`) includes:
- **Dismantling Fixed Effects**: Implements Entity (Firm) and Time (Year) fixed effects via within-group demeaning.
- **Double Selection**: Uses Lasso-based variable selection to identify relevant confounders from 33 initial control variables.
- **Hyperparameter Tuning**: Automated `GridSearchCV` for Random Forest nuisance models to ensure optimal predictive performance.
- **DML Implementation**: Utilizes the `DoubleML` library with 5-fold cross-fitting and 10 repetitions ($n\_rep=10$) for stability.
- **Orthogonality Diagnostics**: Calculates Pearson correlation between residuals to verify the satisfaction of the DML orthogonality condition.

##  Prerequisites
- Python 3.8+
- Required Libraries:
  ```bash
  pip install pandas numpy scikit-learn doubleml scipy openpyxl
