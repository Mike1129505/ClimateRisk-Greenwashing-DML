import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from doubleml import DoubleMLData, DoubleMLPLR
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

def tune_and_check(df, X_vars, T_vars, Y_vars, use_variable_selection=True,
                   rf_param_grid=None, lasso_cv=5, random_state=42):
    """
    Tune Random Forest hyperparameters by comparing cross-validated R²,
    then run DML with RF (using best params) and Lasso, and report all metrics.
    """
    if rf_param_grid is None:
        rf_param_grid = {
            'n_estimators': [300, 500, 1000],
            'max_depth': [5, 7, 10],
            'min_samples_leaf': [5, 10, 20]
        }

    # 1. Clean data
    df = df.copy()
    all_cols = X_vars + T_vars + Y_vars
    # --- Automatically identify Fixed Effects columns (Firm ID and Year) ---
    fe_cols = ['Stkcd', 'Year']
    df = df.dropna(subset=all_cols + fe_cols).reset_index(drop=True)

    # ---------- 2. Add Fixed Effects (Within-transformation) ----------
    # Implement Entity and Time Fixed Effects by performing within-group demeaning
    for col in all_cols:
        # Individual Fixed Effects (Entity FE)
        df[col] = df[col] - df.groupby('Stkcd')[col].transform('mean')
        # Time Fixed Effects (Year FE)
        df[col] = df[col] - df.groupby('Year')[col].transform('mean')
    # --------------------------------------------------

    # 3. Double Selection (Lasso)
    if use_variable_selection:
        from sklearn.linear_model import Lasso
        def select_lasso(Y, X, alpha=0.01):
            lasso = Lasso(alpha=alpha, max_iter=10000, random_state=random_state)
            lasso.fit(X, Y)
            return X.columns[np.abs(lasso.coef_) > 1e-6].tolist()

        X_selected = []
        for y in Y_vars:
            X_selected += select_lasso(df[y], df[X_vars])
        for t in T_vars:
            X_selected += select_lasso(df[t], df[X_vars])
        X_selected = list(set(X_selected))
        if len(X_selected) == 0:
            X_selected = X_vars
        X_use = X_selected
    else:
        X_use = X_vars

    # 4. Standardize controls (Commented out as data is already standardized)
    # scaler = StandardScaler()
    # df[X_use] = scaler.fit_transform(df[X_use])

    # Use the processed df for the following steps
    df_dml = df.copy()

    # 5. Tune Random Forest hyperparameters
    print("Tuning Random Forest hyperparameters on E[Greenwashing|X]...")
    rf_base = RandomForestRegressor(random_state=random_state)
    grid_search = GridSearchCV(rf_base, rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(df_dml[X_use], df_dml['GW_Score'])
    best_rf_params = grid_search.best_params_
    best_rf_score = grid_search.best_score_
    print(f"Best RF params: {best_rf_params} (CV R² = {best_rf_score:.4f})")

    # 6. DML with RF (using best params) and Lasso
    def get_rf_best():
        return RandomForestRegressor(**best_rf_params, random_state=random_state)

    def get_lasso():
        return LassoCV(cv=lasso_cv, max_iter=10000, random_state=random_state)

    results_rf = {}
    results_lasso = {}

    for t in T_vars:
        for y in Y_vars:
            dml_data = DoubleMLData(df_dml, y_col=y, d_cols=t, x_cols=X_use)

            # --- Random Forest DML (n_rep=10) ---
            dml_rf = DoubleMLPLR(dml_data, get_rf_best(), get_rf_best(), n_folds=5, n_rep=10)
            dml_rf.fit()

            # Calculate orthogonality residuals and R2
            try:
                # 优先尝试自动获取
                res = dml_rf.residuals if hasattr(dml_rf, 'residuals') else dml_rf.residuals_
                if res is not None:
                    y_res_rf = res['y_res'] if isinstance(res, dict) else res[:, 0, 0]
                    d_res_rf = res['d_res'] if isinstance(res, dict) else res[:, 0, 1]
                    rho_rf, rho_p_rf = pearsonr(y_res_rf, d_res_rf)
                else:
                    raise ValueError
            except:
                # 手动逻辑
                X_arr, y_arr, t_arr = df_dml[X_use].values, df_dml[y].values, df_dml[t].values
                rf_y = get_rf_best().fit(X_arr, y_arr)
                rf_t = get_rf_best().fit(X_arr, t_arr)
                rho_rf, rho_p_rf = pearsonr(y_arr - rf_y.predict(X_arr), t_arr - rf_t.predict(X_arr))

            r2_y_rf = cross_val_score(get_rf_best(), df_dml[X_use], df_dml[y], cv=5).mean()

            results_rf.update({
                f'{y}_{t}_coef': dml_rf.coef[0], f'{y}_{t}_pval': dml_rf.pval[0],
                f'{y}_{t}_rho': rho_rf, f'{y}_{t}_rho_p': rho_p_rf, f'{y}_{t}_r2_y': r2_y_rf
            })

            # --- Lasso DML (n_rep=10) ---
            dml_lasso = DoubleMLPLR(dml_data, get_lasso(), get_lasso(), n_folds=5, n_rep=10)
            dml_lasso.fit()

            try:
                # 优先尝试自动获取
                res_l = dml_lasso.residuals if hasattr(dml_lasso, 'residuals') else dml_lasso.residuals_
                if res_l is not None:
                    y_res_l = res_l['y_res'] if isinstance(res_l, dict) else res_l[:, 0, 0]
                    d_res_l = res_l['d_res'] if isinstance(res_l, dict) else res_l[:, 0, 1]
                    rho_lasso, rho_p_lasso = pearsonr(y_res_l, d_res_l)
                else:
                    raise ValueError
            except:
                # 手动逻辑
                X_arr, y_arr, t_arr = df_dml[X_use].values, df_dml[y].values, df_dml[t].values
                lasso_y = get_lasso().fit(X_arr, y_arr)
                lasso_t = get_lasso().fit(X_arr, t_arr)
                rho_lasso, rho_p_lasso = pearsonr(y_arr - lasso_y.predict(X_arr), t_arr - lasso_t.predict(X_arr))

            r2_y_lasso = cross_val_score(get_lasso(), df_dml[X_use], df_dml[y], cv=5).mean()

            results_lasso.update({
                f'{y}_{t}_coef': dml_lasso.coef[0], f'{y}_{t}_pval': dml_lasso.pval[0],
                f'{y}_{t}_rho': rho_lasso, f'{y}_{t}_rho_p': rho_p_lasso, f'{y}_{t}_r2_y': r2_y_lasso
            })

    return {
        'best_rf_params': best_rf_params,
        'best_rf_cv_r2': best_rf_score,
        'results_rf': results_rf,
        'results_lasso': results_lasso,
        'X_vars_used': X_use
    }


# =================================================================
# Original Usage Main Block
# =================================================================
if __name__ == "__main__":
    input_file = #please add route

    # Load dataset
    df = pd.read_excel(input_file, engine='openpyxl')

    X_vars = [
        'Size', 'Lev', 'ROA', 'Digitalization', 'Cashflow', 'Liquid', 'Quick', 'Tang_Ratio',
        'Intang_Ratio', 'Growth', 'Asset_Growth', 'Fin_Constraint', 'TobinQ', 'BM',
        'Analyst', 'Board_Size', 'Indep_Ratio', 'Inst_Hold', 'Mng_Hold', 'Fin_Back',
        'Oversea_Back', 'Big4', 'Wind_E_Score', 'TFP', 'EPS', 'RD_Invest',
        'Subsidy', 'HHI', 'IC_Quality', 'Media_Neg', 'Wage', 'List_Age', 'Firm_Age'
    ]
    T_vars = ['Physical_Risk', 'Trans_Risk']
    Y_vars = ['GW_Score', 'Real_Efficiency'] #Change to DGW for robustness

    # Run analysis with Fixed Effects and n_rep=10
    result = tune_and_check(df, X_vars, T_vars, Y_vars, use_variable_selection=True)

    print("\n=== Best Random Forest Hyperparameters ===")
    print(result['best_rf_params'])
    print(f"Best cross-validated R² (on E[Greenwashing|X]): {result['best_rf_cv_r2']:.4f}")

    print("\n=== RF DML Results ===")
    for k, v in result['results_rf'].items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== Lasso DML Results ===")
    for k, v in result['results_lasso'].items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")