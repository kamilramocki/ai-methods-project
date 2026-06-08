import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
# regresory
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
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
    models_list = ["Linear Regression", "Random Forest", "Gradient Boosting", "KNN", "Decision Tree"]

    all_results = []
    results_matrix = np.zeros((2, 15, 5, 10, 4)) # 2 zbiory danych, 15 miast, 5 modeli, 10 foldów, 4 metryki

    features_json_path = os.path.join(PROCESSED_DIR, "top_80_features.json")
    if not os.path.exists(features_json_path):
        print(f"Nie znaleziono pliku z cechami: {features_json_path}")
        return

    with open(features_json_path, "r", encoding="utf-8") as f:
        top_features_dict = json.load(f)

    rkf = RepeatedKFold(n_splits=5, n_repeats=2)

    for d_idx, (dataset_name, (filename, mapping_filename)) in enumerate(datasets.items()):
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
            print(f"\nPrzetwarzanie miasta: {city_name} (id: {city_id})\n")

            city_df = df[df["city"] == city_id].copy()
            top_k80_cols = top_features_dict.get(dataset_name, {}).get(str(int(city_id)))

            X = city_df[top_k80_cols]
            y = city_df[TARGET_COL]

            print(f"Uczenie modeli dla miasta {city_name}")

            fold_metrics = {m_name: {"mae": [], "mse": [], "rmse": [], "r2": []} for m_name in models_list}
            for fold_idx, (train_index, test_index) in enumerate(rkf.split(X, y)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                scaler = StandardScaler()

                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "KNN": KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                }

                for m_idx, (model_name, model) in enumerate(models.items()):
                    try:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)

                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)

                        results_matrix[d_idx, city_id, m_idx, fold_idx, 0] = mae
                        results_matrix[d_idx, city_id, m_idx, fold_idx, 1] = mse
                        results_matrix[d_idx, city_id, m_idx, fold_idx, 2] = rmse
                        results_matrix[d_idx, city_id, m_idx, fold_idx, 3] = r2

                        print(f"    - {model_name:<20} | MAE: {mae:10.2f} | MSE: {mse:10.2f} | RMSE: {rmse:10.2f} | R2: {r2:7.4f}")

                        fold_metrics[model_name]["mae"].append(mae)
                        fold_metrics[model_name]["mse"].append(mse)
                        fold_metrics[model_name]["rmse"].append(rmse)
                        fold_metrics[model_name]["r2"].append(r2)

                    except Exception as e:
                        print(f"    - [BŁĄD] Model {model_name} zgłosił błąd: {e}")

            print(f"\n--- Podsumowanie (Średnie wyniki z 10 foldów) dla miasta {city_name} ---")
            for m_idx, model_name in enumerate(models_list):
                if fold_metrics[model_name]["mae"]:  # Sprawdzamy czy lista nie jest pusta (w razie błędów)
                    mean_mae = np.mean(fold_metrics[model_name]["mae"])
                    mean_mse = np.mean(fold_metrics[model_name]["mse"])
                    mean_rmse = np.mean(fold_metrics[model_name]["rmse"])
                    mean_r2 = np.mean(fold_metrics[model_name]["r2"])

                    print(f"    - {model_name:<20} | MAE: {mean_mae:10.2f} | RMSE: {mean_rmse:10.2f} | R2: {mean_r2:7.4f}")

                    # Zapisujemy uśrednione statystyki miasta do pliku CSV
                    all_results.append({
                        "Dataset": dataset_name,
                        "City": city_name,
                        "Model": model_name,
                        "MAE": mean_mae,
                        "MSE": mean_mse,
                        "RMSE": mean_rmse,
                        "R2": mean_r2
                    })

    np.save("cv_results_matrix.npy", results_matrix)
    print("\nPełna macierz wyników CV została zapisana do: cv_results_matrix.npy")
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
        print(f"\nWyniki zapisano do pliku: {OUTPUT_CSV_PATH}")
    else:
        print("\nBrak wyników do zapisania.")

if __name__ == "__main__":
    main()
