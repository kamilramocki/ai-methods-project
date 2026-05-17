import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ścieżki
DATA_DIR = "data"
OUTPUT_DIR_GRAPHS = "output_graphs"
os.makedirs(OUTPUT_DIR_GRAPHS, exist_ok=True)

def main():
    # 1. zbieranie i łączenie plików csv
    print("skanowanie i łączenie plików")
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    rent_files = [f for f in all_files if "rent" in os.path.basename(f)]
    sale_files = [f for f in all_files if "rent" not in os.path.basename(f)]

    df_rent = pd.concat([pd.read_csv(f) for f in rent_files], ignore_index=True) if rent_files else pd.DataFrame()
    df_sale = pd.concat([pd.read_csv(f) for f in sale_files], ignore_index=True) if sale_files else pd.DataFrame()

    print(f"łącznie ofert - Kupno: {len(df_sale)}, Wynajem: {len(df_rent)}")

    # 2. kluczowe kolumny
    core_features = ['squareMeters', 'rooms', 'floor', 'buildYear']
    target_col = 'price'
    all_needed_cols = ['city', target_col] + core_features

    df_sale = df_sale[all_needed_cols].dropna(subset=['city', 'price'])
    df_rent = df_rent[all_needed_cols].dropna(subset=['city', 'price'])

    # 3. czyszczenie braków
    for df in [df_sale, df_rent]:
        if df.empty: continue
        for col in ['floor', 'buildYear', 'rooms', 'squareMeters']:
            # mediana w obrębie danego miasta
            df[col] = df.groupby('city')[col].transform(lambda x: x.fillna(x.median()))

    # 4. generowanie histogramów
    def generate_histograms(df_s, df_r, city_name="Wszystkie miasta"):
        plt.figure(figsize=(14, 5))
        
        # kupno
        plt.subplot(1, 2, 1)
        if not df_s.empty:
            plt.hist(df_s['price'], bins=50, color='royalblue', alpha=0.7, edgecolor='black')
        plt.title(f'Rozkład cen Kupna - {city_name}')
        plt.xlabel('Cena (PLN)')
        plt.ylabel('Liczba ofert')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # wynajem
        plt.subplot(1, 2, 2)
        if not df_r.empty:
            plt.hist(df_r['price'], bins=50, color='seagreen', alpha=0.7, edgecolor='black')
        plt.title(f'Rozkład cen Wynajmu - {city_name}')
        plt.xlabel('Cena (PLN / miesiąc)')
        plt.ylabel('Liczba ofert')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        filename = f"histogram_{city_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(OUTPUT_DIR_GRAPHS, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"zapisano wykres: {filepath}")

    # generowanie globalnego histogramu
    print("generowanie histogramów")
    generate_histograms(df_sale, df_rent)

    # 5. podział na miasta
    miasta_sale = set(df_sale['city'].unique()) if not df_sale.empty else set()
    miasta_rent = set(df_rent['city'].unique()) if not df_rent.empty else set()
    miasta = miasta_sale.intersection(miasta_rent)

    datasets = {}
    for miasto in miasta:
        # filtrowanie
        sale_city = df_sale[df_sale['city'] == miasto]
        rent_city = df_rent[df_rent['city'] == miasto]
        
        # generowanie histogramów
        generate_histograms(sale_city, rent_city, city_name=miasto.capitalize())
        
        # konwersja do numpy
        X_sale = sale_city[core_features].to_numpy()
        y_sale = sale_city[target_col].to_numpy()
        
        X_rent = rent_city[core_features].to_numpy()
        y_rent = rent_city[target_col].to_numpy()
        
        
        # zapis struktur do słownika
        datasets[miasto] = {
            'X_sale': X_sale, 'y_sale': y_sale,
            'X_rent': X_rent, 'y_rent': y_rent
        }

    return datasets

if __name__ == "__main__":
    main()
