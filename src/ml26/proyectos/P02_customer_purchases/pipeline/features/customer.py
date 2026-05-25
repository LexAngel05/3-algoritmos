"""
Ingeniería de features por cliente.

Modifica extract_customer_features para agregar las estadísticas que
quieras calcular por cliente. El resultado se persiste en
customer_features.csv y se reutiliza al momento de predecir sobre ítems nuevos
(donde no hay historial de compra).
"""

import os

import numpy as np
import pandas as pd

from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    DATA_COLLECTED_AT,
    DATA_DIR,
)


def extract_customer_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features agregadas por cliente a partir del historial de compras.

    Esta función se llama UNA SOLA VEZ sobre los datos de entrenamiento.
    El resultado se guarda en customer_features.csv y se reutiliza en test
    porque el conjunto de test no tiene historial de compra para agregar.

    Parameters
    ----------
    train_df : DataFrame completo de compras de entrenamiento (solo positivos).

    Returns
    -------
    pd.DataFrame con una fila por customer_id.
    """
    df = train_df.copy()
    df["item_release_date"] = pd.to_datetime(df["item_release_date"])
    df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"])
    df["customer_date_of_birth"] = pd.to_datetime(df["customer_date_of_birth"])
    df["customer_signup_date"] = pd.to_datetime(df["customer_signup_date"])

    group = df.groupby("customer_id")

    today_ts = pd.to_datetime(DATA_COLLECTED_AT)

    # ── Ejemplo: edad del cliente en años ──────────────────────────────────
    customer_age_years = (
        today_ts - group["customer_date_of_birth"].first()
    ).dt.days // 365

    # ── Ejemplo: antigüedad en la plataforma en meses ──────────────────────
    customer_tenure_months = (
        today_ts - group["customer_signup_date"].first()
    ).dt.days // 30

    # ── TODO: agrega aquí tus propias features ─────────────────────────────
    # Precio promedio y gasto total por cliente
    customer_avg_price = group["item_price"].mean()
    customer_total_spend = group["item_price"].sum()
    customer_std_price = group["item_price"].std().fillna(0)

    # Precio máximo y mínimo que ha pagado el cliente
    customer_max_price = group["item_price"].max()
    customer_min_price = group["item_price"].min()

    # Numero de compras en el historial
    customer_num_purchases = group["purchase_id"].count()

    # Cuantas categorías distintas ha comprado
    customer_n_categories = group["item_category"].nunique()
    
    # Top 3 categorias más compradas
    def top_cat(x, n):
        counts = x.value_counts()
        return counts.index[n] if len(counts) > n else counts.index[0]
    
    customer_top_1_cat = group["item_category"].agg(lambda x: top_cat(x, 0))
    customer_top_2_cat = group["item_category"].agg(lambda x: top_cat(x, 1))
    customer_top_3_cat = group["item_category"].agg(lambda x: top_cat(x, 2))

    # Dispositivo preferido de compra
    customer_preferred_device = group["purchase_device"].agg(lambda x: x.mode().iloc[0])

    # Género — rellenar nulos con "unknown"
    customer_gender = group["customer_gender"].first().fillna("unknown")

    # Promedio de vistas antes de comprar
    customer_avg_views = group["customer_item_views"].mean()


    # ── Construir DataFrame final ───────────────────────────────────────────

    # NOTA: para los features del cliente usa la convencion customer_[FEATURE_NAME] ya que esto facilitará el trabajo del preprocessing
    customer_feat = pd.concat(
        {
            "customer_id": group["customer_id"].first(),
            "customer_age_years": customer_age_years,
            "customer_tenure_months": customer_tenure_months,
            "customer_avg_price": customer_avg_price,
            "customer_total_spend": customer_total_spend,
            "customer_std_price": customer_std_price,
            "customer_num_purchases": customer_num_purchases,
            "customer_n_categories": customer_n_categories,
            "customer_top_1_cat": customer_top_1_cat,
            "customer_top_2_cat": customer_top_2_cat,
            "customer_top_3_cat": customer_top_3_cat,
            "customer_preferred_device": customer_preferred_device,
            "customer_gender": customer_gender,
            "customer_avg_views": customer_avg_views,
            "customer_max_price": customer_max_price,
            "customer_min_price": customer_min_price,
            # Agrega aquí las features que calculaste arriba, por ejemplo:
            # "customer_avg_price": customer_avg_price,
        },
        axis=1,
    ).reset_index(drop=True)

    # Persistir — read_test_data() carga este archivo en lugar de recomputar
    save_path = os.path.abspath(os.path.join(DATA_DIR, "customer_features.csv"))
    customer_feat.to_csv(save_path, index=False)
    print(f"Customer features saved -> {save_path}")
    return customer_feat
