from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
def evaluate_predictions(y_true, preds, name="Model"):
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mape_val = mape(y_true, preds)

    print(f"\n={name} ")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape_val:.2f}%")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape_val}
