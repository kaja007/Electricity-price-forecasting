import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# ==============================
# XGBOOST TRAINING
# ==============================

def train_xgboost(X_train, y_train, n_trials=40):

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'random_state': 42,
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for tr_idx, val_idx in tscv.split(X_train):
            model = xgb.XGBRegressor(**params, tree_method='hist', verbosity=0)

            model.fit(
                X_train.iloc[tr_idx], y_train.iloc[tr_idx],
                eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                verbose=False
            )

            preds = model.predict(X_train.iloc[val_idx])
            score = np.mean(np.abs(y_train.iloc[val_idx] - preds))  # MAE
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_model = xgb.XGBRegressor(
        **study.best_params,
        tree_method='hist',
        random_state=42
    )

    best_model.fit(X_train, y_train)

    return best_model, study.best_params


# ==============================
# LIGHTGBM TRAINING
# ==============================

def train_lightgbm(X_train, y_train, n_trials=40):

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'random_state': 42,
            'verbose': -1,
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for tr_idx, val_idx in tscv.split(X_train):
            model = lgb.LGBMRegressor(**params)

            model.fit(
                X_train.iloc[tr_idx], y_train.iloc[tr_idx],
                eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=-1)
                ]
            )

            preds = model.predict(X_train.iloc[val_idx])
            score = np.mean(np.abs(y_train.iloc[val_idx] - preds))  # MAE
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_model = lgb.LGBMRegressor(
        **study.best_params,
        random_state=42,
        verbose=-1
    )

    best_model.fit(X_train, y_train)

    return best_model, study.best_params


# ==============================
# TRAINING PIPELINE (COMPILE SCRIPT)
# ==============================

def train_models(df, feature_cols, target_col):
    """
    Complete training pipeline (NO evaluation here)
    """

    # ------------------------------
    # Train-Test Split
    # ------------------------------
    train = df[df['datetime'] < '2023-10-01']
    test  = df[df['datetime'] >= '2023-10-01']

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_test  = test[feature_cols]
    y_test  = test[target_col]

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # ------------------------------
    # Train Models
    # ------------------------------
    print("\nTraining XGBoost...")
    xgb_model, xgb_params = train_xgboost(X_train, y_train)

    print("\nTraining LightGBM...")
    lgb_model, lgb_params = train_lightgbm(X_train, y_train)

    # ------------------------------
    # Predictions (no evaluation)
    # ------------------------------
    pred_xgb = xgb_model.predict(X_test)
    pred_lgb = lgb_model.predict(X_test)

    # ------------------------------
    # Return everything
    # ------------------------------
    return {
        "models": {
            "xgb": xgb_model,
            "lgb": lgb_model
        },
        "params": {
            "xgb": xgb_params,
            "lgb": lgb_params
        },
        "predictions": {
            "xgb": pred_xgb,
            "lgb": pred_lgb
        },
        "y_test": y_test
    }