import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml26.proyectos.P02_customer_purchases.pipeline import read_test_data
from ml26.proyectos.P02_customer_purchases.pipeline.io import read_csv

CURRENT_FILE = Path(__file__).resolve()

RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MODELS_DIR = CURRENT_FILE.parent / "trained_models"

WINNING_XGB_MODEL = "xgb_lbfgs_1000_20260601_195642"
WINNING_RF_MODEL = "rf_lbfgs_1000_20260601_195638"


def run_inference(model_path: str, X, ids=None):
    """
    Genera predicciones usando un solo modelo guardado.
    """
    full_path = MODELS_DIR / f"{model_path}/model.pkl"  # Ruta del archivo model.pkl dentro de trained_models.
    print(f"Loading model from {full_path}")

    model = joblib.load(full_path)  # Carga el modelo entrenado desde disco.
    preds = model.predict(X)  # Predice la clase final 0 o 1.
    probs = model.predict_proba(X)[:, 1]  # Calcula la probabilidad de compra.

    return pd.DataFrame(
        {"ID": ids if ids is not None else X.index, "pred": preds, "proba": probs}
    )


def run_inference_random(X, ids=None):
    """
    Genera predicciones aleatorias como baseline de comparacion.
    """
    probs = np.random.uniform(0, 1, size=len(X))  # Genera probabilidades aleatorias entre 0 y 1.
    preds = (probs >= 0.5).astype(int)  # Convierte las probabilidades aleatorias a 0 o 1.
    return pd.DataFrame(
        {"ID": ids if ids is not None else X.index, "pred": preds, "proba": probs}
    )


def run_inference_weighted_ensemble(
    xgb_model_path: str,
    rf_model_path: str,
    X,
    ids=None,
    xgb_weight: float = 0.40,
    rf_weight: float = 0.60,
    threshold: float = 0.50,
):
    """
    Combina XGBoost y Random Forest para reproducir la submission ganadora.

    La configuracion 40% XGBoost + 60% Random Forest con threshold 0.50 obtuvo
    0.97218 en Kaggle.
    """
    xgb_path = MODELS_DIR / f"{xgb_model_path}/model.pkl"  # Ruta del modelo XGBoost guardado.
    rf_path = MODELS_DIR / f"{rf_model_path}/model.pkl"  # Ruta del modelo Random Forest guardado.
    print(f"Loading XGBoost model from {xgb_path}")
    print(f"Loading Random Forest model from {rf_path}")

    xgb_model = joblib.load(xgb_path)  # Carga el modelo XGBoost entrenado.
    rf_model = joblib.load(rf_path)  # Carga el modelo Random Forest entrenado.

    xgb_probs = xgb_model.predict_proba(X)[:, 1]  # Probabilidad de compra segun XGBoost.
    rf_probs = rf_model.predict_proba(X)[:, 1]  # Probabilidad de compra segun Random Forest.
    ensemble_probs = (xgb_weight * xgb_probs) + (rf_weight * rf_probs)  # Promedio ponderado de ambos modelos.
    preds = (ensemble_probs >= threshold).astype(int)  # Convierte la probabilidad combinada en 0 o 1.

    return pd.DataFrame(
        {"ID": ids if ids is not None else X.index, "pred": preds, "proba": ensemble_probs}
    )


if __name__ == "__main__":
    test_ids = read_csv("customer_purchases_test")["purchase_id"]  # Lee los IDs que Kaggle espera.
    X = read_test_data()  # Preprocesa el test con el preprocessor guardado.

    xgb_model_folder = WINNING_XGB_MODEL  # Usa el XGBoost que participo en la submission 0.97218.
    rf_model_folder = WINNING_RF_MODEL  # Usa el Random Forest que participo en la submission 0.97218.

    print(f"Usando modelo XGBoost: {xgb_model_folder}")
    print(f"Usando modelo Random Forest: {rf_model_folder}")

    xgb_results = run_inference(xgb_model_folder, X, ids=test_ids)  # Genera predicciones del XGBoost individual.
    xgb_dir = RESULTS_DIR / xgb_model_folder  # Carpeta donde se guarda la salida individual.
    xgb_dir.mkdir(exist_ok=True, parents=True)  # Crea la carpeta si no existe.
    xgb_path = xgb_dir / "predictions.csv"  # Archivo compatible con Kaggle para XGBoost individual.
    xgb_results[["ID", "pred"]].to_csv(xgb_path, index=False)  # Guarda solo ID y pred, que es lo que pide Kaggle.
    print(f"Saved XGBoost predictions to {xgb_path}")

    ensemble_results = run_inference_weighted_ensemble(
        xgb_model_folder,
        rf_model_folder,
        X,
        ids=test_ids,
        xgb_weight=0.40,
        rf_weight=0.60,
        threshold=0.50,
    )  # Genera predicciones del ensamble ganador 40/60.
    ensemble_dir = RESULTS_DIR / "weighted_ensemble_xgb40_rf60"  # Carpeta fija para el ensamble ganador.
    ensemble_dir.mkdir(exist_ok=True, parents=True)  # Crea la carpeta si no existe.
    ensemble_path = ensemble_dir / "weighted_xgb40_rf60_050.csv"  # Archivo final del ensamble ganador.
    ensemble_results[["ID", "pred"]].to_csv(ensemble_path, index=False)  # Guarda solo ID y pred para Kaggle.
    print(f"Saved weighted ensemble predictions to {ensemble_path}")
