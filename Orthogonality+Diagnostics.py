import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from doubleml import DoubleMLData, DoubleMLPLR
from scipy.stats import pearsonr

# ---------- User Configuration ----------
#data_path = '
#input_file = os.path.join(data_path, '.xlsx')
#output_excel = os.path.join(data_path, 'Orthogonality_Diagnostics_with_R2.xlsx')

# ---------- Variable Definitions ----------
T_vars = ['Physical_Risk', 'Trans_Risk']
Y_vars = ['GW_Score', 'Real_Efficiency']
X_vars = [
    'Size', 'Lev', 'ROA', 'Digitalization', 'Cashflow', 'Liquid', 'Quick', 'Tang_Ratio',
    'Intang_Ratio', 'Growth', 'Asset_Growth', 'Fin_Constraint', 'TobinQ', 'BM',
    'Analyst', 'Board_Size', 'Indep_Ratio', 'Inst_Hold', 'Mng_Hold', 'Fin_Back',
    'Oversea_Back', 'Big4', 'Wind_E_Score', 'TFP', 'EPS', 'RD_Invest',
    'Subsidy', 'HHI', 'IC_Quality', 'Media_Neg', 'Wage', 'List_Age', 'Firm_Age'
]
# Fixed Effects identifiers
FE_vars = ['Stkcd', 'Year']

# ---------- Load Data ----------
df = pd.read_excel(input_file, engine='openpyxl')
df = df.dropna(subset=X_vars + T_vars + Y_vars + FE_vars).reset_index(drop=True)

# ---------- IMPLEMENT FIXED EFFECTS (Within-Transformation) ----------
# To satisfy the panel data structure mentioned in the paper, we demean the variables
# This handles Entity (Stkcd) and Time (Year) Fixed Effects
all_study_vars = X_vars + T_vars + Y_vars
for col in all_study_vars:
    # Individual Fixed Effects
    df[col] = df[col] - df.groupby('Stkcd')[col].transform('mean')
    # Time Fixed Effects
    df[col] = df[col] - df.groupby('Year')[col].transform('mean')


# --------------------------------------------------

# Standardize control variables (Commented out as data is already standardized, if use raw data, cancel comment)
# scaler = StandardScaler()
# df[X_vars] = scaler.fit_transform(df[X_vars])


# ---------- Variable Selection (Double Selection) ----------
def variable_selection_lasso(y, X, alpha=0.01):
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    lasso.fit(X, y)
    return X.columns[np.abs(lasso.coef_) > 1e-6].tolist()


print("Performing Double Selection (Lasso, alpha=0.01)...")
selected_for_y = {yv: variable_selection_lasso(df[yv], df[X_vars]) for yv in Y_vars}
selected_for_t = {tv: variable_selection_lasso(df[tv], df[X_vars]) for tv in T_vars}
all_selected = list(set().union(*selected_for_y.values(), *selected_for_t.values()))
if not all_selected:
    all_selected = X_vars[:]
X_use = all_selected

# Prepare data for DML (using processed variables)
df_X = df[X_use + T_vars + Y_vars].copy()


# Standardize selected variables (Commented out as data is already standardized)
# scaler2 = StandardScaler()
# df_X[X_use] = scaler2.fit_transform(df_X[X_use])


# ---------- Learner Factories ----------
def get_rf_grid():
    param_grid = {
        'n_estimators': [300, 500, 1000],
        'max_depth': [5, 7, 10],
        'min_samples_leaf': [5, 10, 20]
    }
    rf = RandomForestRegressor(random_state=42)
    return GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)


def get_lasso():
    return LassoCV(cv=5, max_iter=10000, random_state=42)


# ---------- Collect Orthogonality Diagnostics + CV R² ----------
ortho_results = []

for t in T_vars:
    for y in Y_vars:
        for learner_name, learner_func in [('RF', get_rf_grid), ('Lasso', get_lasso)]:
            print(f"Running DML for Y={y}, T={t}, Learner={learner_name}...")
            dml_data = DoubleMLData(df_X, y_col=y, d_cols=t, x_cols=X_use)
            ml_l = learner_func()
            ml_m = learner_func()

            # n_rep=10 as specified in your paper's methodology
            dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=5, n_rep=10)
            dml_plr.fit()

            # Retrieve residuals for orthogonality check
            try:
                res = dml_plr.residuals if hasattr(dml_plr, 'residuals') else dml_plr.residuals_
                y_res = res['y_res'].values
                d_res = res['d_res'].values
            except Exception as e:
                # Manual calculation backup
                X_arr = df_X[X_use].values
                y_arr = df_X[y].values
                t_arr = df_X[t].values
                if learner_name == 'RF':
                    ml_y = RandomForestRegressor(n_estimators=500, max_depth=7, random_state=42).fit(X_arr, y_arr)
                    ml_t = RandomForestRegressor(n_estimators=500, max_depth=7, random_state=42).fit(X_arr, t_arr)
                else:
                    ml_y = LassoCV(cv=5, max_iter=10000, random_state=42).fit(X_arr, y_arr)
                    ml_t = LassoCV(cv=5, max_iter=10000, random_state=42).fit(X_arr, t_arr)
                y_res = y_arr - ml_y.predict(X_arr)
                d_res = t_arr - ml_t.predict(X_arr)

            rho, rho_p = pearsonr(y_res, d_res)

            # Calculate first-stage CV R² for E[Y|X] and E[T|X]
            if learner_name == 'RF':
                ml_y_cv = RandomForestRegressor(n_estimators=500, max_depth=7, random_state=42)
                ml_t_cv = RandomForestRegressor(n_estimators=500, max_depth=7, random_state=42)
            else:
                ml_y_cv = LassoCV(cv=5, max_iter=10000, random_state=42)
                ml_t_cv = LassoCV(cv=5, max_iter=10000, random_state=42)

            # 5-fold CV R²
            r2_y = cross_val_score(ml_y_cv, df_X[X_use].values, df_X[y].values, cv=5, scoring='r2').mean()
            r2_t = cross_val_score(ml_t_cv, df_X[X_use].values, df_X[t].values, cv=5, scoring='r2').mean()

            ortho_results.append({
                'Outcome (Y)': y,
                'Treatment (T)': t,
                'Learner': learner_name,
                'Orthogonal_Rho': rho,
                'Orthogonal_Pval': rho_p,
                'CV_R2_E[Y|X]': r2_y,
                'CV_R2_E[T|X]': r2_t
            })

# ---------- Output Results ----------
df_ortho = pd.DataFrame(ortho_results)
print("\n=== Orthogonality Diagnostics with CV R² (Table 3 Format) ===")
print(df_ortho.to_string(index=False))
df_ortho.to_excel(output_excel, index=False)
print(f"\nOrthogonality diagnostics saved to: {output_excel}")