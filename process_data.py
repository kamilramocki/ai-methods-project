import os
import glob
import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data"
OUTPUT_DIR = "processed"

def process_dataset(df, name):
    
    # 1. Usunięcie kolumny 'id', która nie ma wartości predykcyjnej dla ML
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # wykrywanie kolumn zawierających jakiekolwiek wartości tekstowe
    categorical_cols = []
    for col in df.columns:
        non_null_vals = df[col].dropna()
        if not non_null_vals.empty and any(isinstance(val, str) for val in non_null_vals):
            categorical_cols.append(col)
            # standaryzacja stringów w celu spójnego mapowania
            df[col] = df[col].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)

    # 2. zamiana zmiennych tekstowych na liczbowe (kategoryzacja) przy użyciu dynamicznych słowników
    print("Dynamiczne mapowanie zmiennych tekstowych na liczbowe...")
    for col in categorical_cols:
        # pobieramy unikalne wartości tekstowe, sortujemy je dla spójnego porządku
        unique_vals = sorted(df[col].dropna().unique())
        
        # tworzymy mapowanie dynamiczne
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        
        # mapujemy wartości
        df[col] = df[col].map(mapping)

    # 3. uzupełnianie brakujących wartości (Imputacja za pomocą SimpleImputer)
    numerical_cols = [c for c in df.columns if c not in categorical_cols]
    
    # dla zmiennych kategorycznych używamy najczęściej występującej wartości (most_frequent)
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        
    # dla zmiennych numerycznych używamy mediany (median)
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='median')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # konwersja typów: kategoryczne kolumny na int (by nie były float po imputacji)
    for col in categorical_cols:
        df[col] = df[col].astype(int)
    
    return df


def main():
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    # filtrowanie plików wejściowych (odrzucamy już przetworzone, jeśli istnieją)
    input_files = [f for f in all_files if "processed" not in os.path.basename(f)]
    
    rent_files = [f for f in input_files if "rent" in os.path.basename(f)]
    sale_files = [f for f in input_files if "rent" not in os.path.basename(f)]
    
    print(f"Znaleziono plików dla Kupna: {len(sale_files)}")
    print(f"Znaleziono plików dla Wynajmu: {len(rent_files)}")
    
    # łączenie w osobne DataFrame'y
    if sale_files:
        df_sale = pd.concat([pd.read_csv(f) for f in sale_files], ignore_index=True)
        df_sale_processed = process_dataset(df_sale, "Kupno (Sale)")
        
        output_sale_path = os.path.join(OUTPUT_DIR, "apartments_sale_processed.csv")
        df_sale_processed.to_csv(output_sale_path, index=False)
        print(f"Zapisano przetworzony zbiór Kupna do: {output_sale_path}")
    else:
        print("Brak plików dla ofert Kupna.")

    if rent_files:
        df_rent = pd.concat([pd.read_csv(f) for f in rent_files], ignore_index=True)
        df_rent_processed = process_dataset(df_rent, "Wynajem (Rent)")
        
        output_rent_path = os.path.join(OUTPUT_DIR, "apartments_rent_processed.csv")
        df_rent_processed.to_csv(output_rent_path, index=False)
        print(f"Zapisano przetworzony zbiór Wynajmu do: {output_rent_path}")
    else:
        print("Brak plików dla ofert Wynajmu.")

    print("\nPrzetwarzanie zakończone sukcesem!")

if __name__ == "__main__":
    main()
