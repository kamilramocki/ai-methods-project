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
        "Kupno (Sale)": ("apartments_sale_processed.csv", "apartments_sale_mapping.json"),
        "Wynajem (Rent)": ("apartments_rent_processed.csv", "apartments_rent_mapping.json"),
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

    with open(features_json_path, "r", encoding="utf-8") as f:
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
            with open(mapping_filepath, "r", encoding="utf-8") as f:
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


            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            print(f"Uczenie modeli dla miasta {city_name}")
            for model_name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    # Obliczanie metryk oceny regresji:
                    # - MAE (Średni błąd bezwzględny) - średnia różnica w jednostkach ceny
                    # - MSE (Średni błąd kwadratowy) - mocniej karze duże błędy
                    # - RMSE (Pierwiastek błędu średniokwadratowego) - w tej samej jednostce co cena, intuicyjny
                    # - R^2 (Współczynnik determinacji) - jak dobrze model tłumaczy wariancję danych (1.0 to ideał)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    print(f"    - {model_name:<20} | MAE: {mae:10.2f} | RMSE: {rmse:10.2f} | R2: {r2:7.4f}")

                    all_results.append({
                        "Dataset": dataset_name,
                        "City": city_name,
                        "Model": model_name,
                        "MAE": mae,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R2": r2
                    })
                except Exception as e:
                    print(f"    - [BŁĄD] Model {model_name} zgłosił błąd: {e}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
        print(f"\nWyniki zapisano do pliku: {OUTPUT_CSV_PATH}")
    else:
        print("\nBrak wyników do zapisania.")

if __name__ == "__main__":
    main()
