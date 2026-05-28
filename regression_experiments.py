import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# regresory
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.stdout.reconfigure(encoding='utf-8')

PROCESSED_DIR = "processed"
OUTPUT_CSV_PATH = "regression_results.csv"
TARGET_COL = "price"

def main():
    datasets = {
        "Kupno (Sale)": ("apartments_sale_processed.csv", "apartments_sale_mapping"),
        "Wynajem (Rent)": ("apartments_rent_processed.csv", "apartments_rent_mapping"),
    }

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "KNN": KNeighborsRegressor(),
        "SVR": SVR(),
    }

    all_results = []

    features_json_path = os.path.join(PROCESSED_DIR, "top_80_features.json")
    if not os.path.exists(features_json_path):
        print(f"Nie znaleziono pliku z cechami: {features_json_path}")
        return

    with open(features_json_path, "r", encoding="uft-8") as f:
        top_features_dict = json.load(f)

    for dataset_name, (filename, mapping_filename) in datasets.items():
        filepath = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(filepath):
            continue

        df = pd.read_csv(filepath)
        print(f"\nRozpoczynam eksperymenty dla: {dataset_name}\n\n")

        city_mapping = {}
        mapping_filepath = os.path.join(PROCESSED_DIR, mapping_filename)
        if os.path.exists(mapping_filepath):
            with open(mapping_filepath, "r", encoding="uft-8") as f:
                full_mapping = json.load(f)
                city_mapping = full_mapping.get("city", {})

        city_ids = sorted(df["city"].dropna().unique())

        for city_id in city_ids:
            city_name = city_mapping.get(str(int(city_id)), str(city_id)).capitalize()
            print(f"\nPrzetwarzanie miasta: {city_name}\n")

            city_df = df[df["city"] == city_id].copy()
            top_k80_cols = top_features_dict.get(dataset_name, {}).get(str(int(city_id)))

            X = city_df[top_k80_cols]
            y = city_df[TARGET_COL]

        # TODO: train_test_split na X i y
        # TODO: Standaryzacja danych
        # TODO: Uczenie i porównywanie regresorów (jeszcze do dogadania czy bierzemy faktycznie te 5 czy jakieś inne)

if __name__ == "__main__":
    main()
