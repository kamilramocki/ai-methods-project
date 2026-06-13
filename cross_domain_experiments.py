import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.stdout.reconfigure(encoding='utf-8')


def main():
    print("--- ROZPOCZĘCIE EKSPERYMENTU CROSS-DOMAIN ---")

    sale_data = pd.read_csv("processed/apartments_sale_processed.csv")
    rent_data = pd.read_csv("processed/apartments_rent_processed.csv")

    print(f"Wczytano zbiór Kupna (Sale): {len(sale_data)} wierszy")
    print(f"Wczytano zbiór Wynajmu (Rent): {len(rent_data)} wierszy")

    with open("processed/top_80_features.json", "r", encoding="utf-8") as f:
        top_features_dict = json.load(f)

    with open("processed/apartments_sale_mapping.json", "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    city_mapping = mapping_data["city"]

    results_list = []

    for city_id in range(15):
        city_id_str = str(city_id)
        city_name = city_mapping[city_id_str].capitalize()

        # 1. Filtrujemy dane dla konkretnego miasta przed normalizacją
        sale_city = sale_data[sale_data['city'] == city_id].copy()
        rent_city = rent_data[rent_data['city'] == city_id].copy()

        if len(sale_city) < 10 or len(rent_city) < 10:
            print(f"Pominięto {city_name} z powodu zbyt małej liczby danych.")
            continue

        print(f"\nObliczenia dla miasta: {city_name} (ID: {city_id})")

        # 2. Lokalna normalizacja
        # Kupno (Sale) dla danego miasta
        y_sale_city = sale_city['price'].values.astype(float)
        y_sale_city -= np.min(y_sale_city)
        y_sale_city /= np.max(y_sale_city)
        sale_city['price'] = y_sale_city

        # Wynajem (Rent) dla danego miasta
        y_rent_city = rent_city['price'].values.astype(float)
        y_rent_city -= np.min(y_rent_city)
        y_rent_city /= np.max(y_rent_city)
        rent_city['price'] = y_rent_city

        # 3. Pobranie cech i zabezpieczenie przed brakiem kolumn
        sale_features = [f for f in top_features_dict["Kupno (Sale)"][city_id_str] if
                         f in rent_city.columns and f in sale_city.columns]
        rent_features = [f for f in top_features_dict["Wynajem (Rent)"][city_id_str] if
                         f in rent_city.columns and f in sale_city.columns]

        # --- KIERUNEK 1: Kupno -> Wynajem ---
        print(" -> Kierunek: Kupno -> Wynajem")
        X_train_1 = sale_city[sale_features]
        y_train_1 = sale_city['price']
        X_test_1 = rent_city[sale_features]
        y_test_1 = rent_city['price']

        scaler_1 = StandardScaler()
        X_train_scaled_1 = scaler_1.fit_transform(X_train_1)
        X_test_scaled_1 = scaler_1.transform(X_test_1)

        rf_model_1 = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_model_1.fit(X_train_scaled_1, y_train_1)
        y_pred_1 = rf_model_1.predict(X_test_scaled_1)

        r2_1 = r2_score(y_test_1, y_pred_1)
        mae_1 = mean_absolute_error(y_test_1, y_pred_1)
        rmse_1 = np.sqrt(mean_squared_error(y_test_1, y_pred_1))
        print(f"    R2 = {r2_1:.4f}, MAE = {mae_1:.4f}, RMSE = {rmse_1:.4f}")

        results_list.append({
            "City": city_name, "Direction": "Buy -> Rent",
            "MAE": mae_1, "RMSE": rmse_1, "R2": r2_1
        })

        # --- KIERUNEK 2: Wynajem -> Kupno ---
        print(" -> Kierunek: Wynajem -> Kupno")
        X_train_2 = rent_city[rent_features]
        y_train_2 = rent_city['price']
        X_test_2 = sale_city[rent_features]
        y_test_2 = sale_city['price']

        scaler_2 = StandardScaler()
        X_train_scaled_2 = scaler_2.fit_transform(X_train_2)
        X_test_scaled_2 = scaler_2.transform(X_test_2)

        rf_model_2 = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_model_2.fit(X_train_scaled_2, y_train_2)
        y_pred_2 = rf_model_2.predict(X_test_scaled_2)

        r2_2 = r2_score(y_test_2, y_pred_2)
        mae_2 = mean_absolute_error(y_test_2, y_pred_2)
        rmse_2 = np.sqrt(mean_squared_error(y_test_2, y_pred_2))
        print(f"    R2 = {r2_2:.4f}, MAE = {mae_2:.4f}, RMSE = {rmse_2:.4f}")

        results_list.append({
            "City": city_name, "Direction": "Rent -> Buy",
            "MAE": mae_2, "RMSE": rmse_2, "R2": r2_2
        })

    results_df = pd.DataFrame(results_list)
    output_path = "cross_domain_results.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nWyniki eksperymentu zapisano w pliku: {output_path}")

    # Wykres porównawczy R2
    buy_to_rent_df = results_df[results_df['Direction'] == 'Buy -> Rent']
    rent_to_buy_df = results_df[results_df['Direction'] == 'Rent -> Buy']

    plt.figure(figsize=(12, 6))
    x = np.arange(len(buy_to_rent_df))
    width = 0.35

    plt.bar(x - width / 2, buy_to_rent_df['R2'], width, label='Kupno -> Wynajem', color='royalblue')
    plt.bar(x + width / 2, rent_to_buy_df['R2'], width, label='Wynajem -> Kupno', color='seagreen')

    plt.xlabel('Miasto')
    plt.ylabel('Wynik R2')
    plt.title('Porównanie wyników R2 w eksperymencie Cross-Domain dla poszczególnych miast')
    plt.xticks(x, buy_to_rent_df['City'], rotation=45)
    plt.ylim(-1, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    os.makedirs("output_graphs", exist_ok=True)
    graph_path = "output_graphs/cross_domain_comparison.png"
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"Zapisano wykres porównawczy w: {graph_path}")


if __name__ == "__main__":
    main()