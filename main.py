# ==========================================
# MAIN PIPELINE SCRIPT
# ==========================================

import warnings
warnings.filterwarnings("ignore")

# Import your modules
from src.preprocess import load_and_preprocess_data
from src.features import create_features
from src.train import train_models
from src.evaluate import evaluate_predictions




DATA_PATH = "data/"   # folder with CSV files




def main():

    

    print("========== STEP 1: DATA PREPROCESSING ==========")
    df = load_and_preprocess_data(DATA_PATH)
    print(f"Data loaded: {df.shape}")

    
    print("\n========== STEP 2: FEATURE ENGINEERING ==========")
    df, feature_cols, target_col = create_features(df)
    print(f"Features created: {len(feature_cols)}")

    
    print("\n========== STEP 3: MODEL TRAINING ==========")
    results = train_models(df, feature_cols, target_col)

    
    print("\n========== STEP 4: MODEL EVALUATION ==========")

    y_test = results["y_test"]

    xgb_metrics = evaluate_predictions(
        y_test,
        results["predictions"]["xgb"],
        "XGBoost"
    )

    lgb_metrics = evaluate_predictions(
        y_test,
        results["predictions"]["lgb"],
        "LightGBM"
    )

    print("\n========== STEP 5: ENSEMBLE ==========")

    pred_xgb = results["predictions"]["xgb"]
    pred_lgb = results["predictions"]["lgb"]

    # Weight by inverse MAE
    w_xgb = 1 / xgb_metrics["MAE"]
    w_lgb = 1 / lgb_metrics["MAE"]

    total = w_xgb + w_lgb
    w_xgb /= total
    w_lgb /= total

    pred_ensemble = (w_xgb * pred_xgb) + (w_lgb * pred_lgb)

    evaluate_predictions(y_test, pred_ensemble, "Ensemble")

    

if __name__ == "__main__":
    main()