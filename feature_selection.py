import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

sys.stdout.reconfigure(encoding='utf-8')

PROCESSED_DIR = "processed"
OUTPUT_DIR = "output_graphs\\p2"
OUTPUT_DIR_CITIES = os.path.join(OUTPUT_DIR, "cities")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_CITIES, exist_ok=True)

TARGET_COL = "price"


def analyze_features(df: pd.DataFrame, dataset_name: str, drop_cols=None):
    print(f"Analiza: {dataset_name}")
    drop_cols = drop_cols or []

    # wczytanie danych
    X = df.drop(columns=[TARGET_COL] + drop_cols, errors="ignore")
    y = df[TARGET_COL]
    feature_names = X.columns.tolist()

    # standaryzacja
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # F-regression
    selector_f = SelectKBest(score_func=f_regression, k="all")
    selector_f.fit(X_scaled, y)
    f_scores = selector_f.scores_
    f_pvalues = selector_f.pvalues_

    # Mutual Information
    mi_scores = mutual_info_regression(X_scaled, y)

    # tabela wyników
    results = pd.DataFrame({
        "feature":   feature_names,
        "f_score":   f_scores,
        "f_pvalue":  f_pvalues,
        "mi_score":  mi_scores,
    })

    # normalizacja wyników do [0, 1]
    results["f_score_norm"] = results["f_score"] / results["f_score"].max()
    results["mi_score_norm"] = results["mi_score"] / results["mi_score"].max()

    # ranking łączony: średnia znormalizowanych wyników
    results["combined_score"] = (results["f_score_norm"] + results["mi_score_norm"]) / 2
    results = results.sort_values("combined_score", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1

    table_data = []
    for _, row in results.iterrows():
        sig = "***" if row["f_pvalue"] < 0.001 else ("**" if row["f_pvalue"] < 0.01 else ("*" if row["f_pvalue"] < 0.05 else ""))
        table_data.append([
            int(row["rank"]),
            row["feature"],
            f"{row['f_score']:.2f}",
            f"{row['f_pvalue']:.4f} {sig}",
            f"{row['mi_score']:.4f}",
            f"{row['combined_score']:.4f}",
        ])

    headers = ["Rank", "Cecha", "F-score", "p-value", "MI score", "Łączny"]
    print(tabulate(table_data, headers=headers, tablefmt="rounded_outline"))


    # Sugestia liczby cech (metoda "łokcia" na skumulowanym MI)
    mi_sorted = results["mi_score"].values
    cumulative = np.cumsum(mi_sorted) / np.sum(mi_sorted)
    k_80 = int(np.argmax(cumulative >= 0.80)) + 1  # minimalna liczba cech pokrywająca 80% MI
    k_90 = int(np.argmax(cumulative >= 0.90)) + 1  # minimalna liczba cech pokrywająca 90% MI

    print(f"\nSugerowana liczba cech:")
    print(f"{k_80} cech pokrywa ~80% łącznego Mutual Information")
    print(f"{k_90} cech pokrywa ~90% łącznego Mutual Information")

    top_features_80 = results["feature"].iloc[:k_80].tolist()
    top_features_90 = results["feature"].iloc[:k_90].tolist()
    print(f"\nTop {k_80} cech (80% MI): {top_features_80}")
    print(f"Top {k_90} cech (90% MI): {top_features_90}")

    return results, k_80, k_90, cumulative


def analyze_features_per_city(df: pd.DataFrame, dataset_name: str, generate_plots: bool = True, output_dir: str = OUTPUT_DIR_CITIES, city_mapping: dict = None):
    if "city" not in df.columns:
        print(f"\nBrak kolumny 'city' w zbiorze: {dataset_name} — pomijam analizę per-miasto.")
        return {}

    city_mapping = city_mapping or {}
    city_results = {}
    city_ids = sorted(df["city"].dropna().unique())

    print(f"\n{'=' * 90}")
    print(f"Analiza per-miasto: {dataset_name}")
    print(f"{'=' * 90}")

    for city_id in city_ids:
        city_df = df[df["city"] == city_id]

        city_name = city_mapping.get(str(int(city_id)), str(city_id))
        city_label = city_name.capitalize()

        label = f"{dataset_name} {city_label}"
        analysis = analyze_features(city_df, label, drop_cols=["city"])

        if analysis is None:
            continue

        results, k_80, k_90, cumulative = analysis
        city_results[city_id] = {
            "results": results,
            "k_80": k_80,
            "k_90": k_90,
        }

        if generate_plots:
            plot_results(results, k_80, k_90, cumulative, label, output_dir=output_dir)

    return city_results


def plot_results(results: pd.DataFrame, k_80: int, k_90: int, cumulative: np.ndarray, dataset_name: str, output_dir: str = OUTPUT_DIR):
    n = len(results)
    features = results["feature"].tolist()
    colors_f  = ["#2563eb" if i < k_80 else "#93c5fd" for i in range(n)]
    colors_mi = ["#16a34a" if i < k_80 else "#86efac" for i in range(n)]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#f8fafc")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    title = f"Analiza istotności cech — {dataset_name}"
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#1e293b", y=0.98)

    # Panel 1: F-score
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(range(n), results["f_score"], color=colors_f, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(features, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_title("F-Score (F-Regression)", fontweight="bold", color="#1e293b")
    ax1.set_xlabel("Wartość F-Score")
    ax1.axhline(k_80 - 0.5, color="#dc2626", linestyle="--", linewidth=1.2, label=f"Top {k_80} (80% MI)")
    ax1.legend(fontsize=8)
    ax1.set_facecolor("#f1f5f9")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    # etykiety wartości
    for bar, val in zip(bars, results["f_score"]):
        ax1.text(bar.get_width() + results["f_score"].max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=7, color="#475569")

    # Panel 2: Mutual Information
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.barh(range(n), results["mi_score"], color=colors_mi, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(features, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_title("Mutual Information Score", fontweight="bold", color="#1e293b")
    ax2.set_xlabel("Wartość MI Score")
    ax2.axhline(k_80 - 0.5, color="#dc2626", linestyle="--", linewidth=1.2,
                label=f"Top {k_80} (80% MI)")
    ax2.legend(fontsize=8)
    ax2.set_facecolor("#f1f5f9")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    for bar, val in zip(bars2, results["mi_score"]):
        ax2.text(bar.get_width() + results["mi_score"].max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=7, color="#475569")

    # Panel 3: Skumulowane MI (wykres łokcia)
    ax3 = fig.add_subplot(gs[1, 0])
    x_vals = np.arange(1, n + 1)
    ax3.plot(x_vals, cumulative * 100, marker="o", markersize=5,
             color="#7c3aed", linewidth=2, label="Skumulowane MI (%)")
    ax3.axhline(80, color="#f59e0b", linestyle="--", linewidth=1.2, label="Próg 80%")
    ax3.axhline(90, color="#dc2626", linestyle="--", linewidth=1.2, label="Próg 90%")
    ax3.axvline(k_80, color="#f59e0b", linestyle=":", linewidth=1.2)
    ax3.axvline(k_90, color="#dc2626", linestyle=":", linewidth=1.2)
    ax3.annotate(f"k={k_80}", xy=(k_80, 80), xytext=(k_80 + 0.3, 75),
                 fontsize=9, color="#f59e0b", fontweight="bold")
    ax3.annotate(f"k={k_90}", xy=(k_90, 90), xytext=(k_90 + 0.3, 85),
                 fontsize=9, color="#dc2626", fontweight="bold")
    ax3.set_title("Skumulowane Mutual Information (metoda łokcia)", fontweight="bold", color="#1e293b")
    ax3.set_xlabel("Liczba wybranych cech (k)")
    ax3.set_ylabel("Skumulowane MI [%]")
    ax3.set_xticks(x_vals)
    ax3.set_xticklabels(x_vals, fontsize=8)
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=8)
    ax3.set_facecolor("#f1f5f9")
    ax3.grid(linestyle="--", alpha=0.4)

    # Panel 4: Ranking łączony
    ax4 = fig.add_subplot(gs[1, 1])
    combined_colors = ["#0f172a" if i < k_80 else "#94a3b8" for i in range(n)]
    bars4 = ax4.barh(range(n), results["combined_score"], color=combined_colors, edgecolor="white", linewidth=0.5)
    ax4.set_yticks(range(n))
    ax4.set_yticklabels(features, fontsize=8)
    ax4.invert_yaxis()
    ax4.set_title("Ranking Łączony (średnia znorm. F-score i MI)", fontweight="bold", color="#1e293b")
    ax4.set_xlabel("Łączny wynik (0–1)")
    ax4.axhline(k_80 - 0.5, color="#dc2626", linestyle="--", linewidth=1.2, label=f"Top {k_80} (80% MI)")
    ax4.legend(fontsize=8)
    ax4.set_facecolor("#f1f5f9")
    ax4.grid(axis="x", linestyle="--", alpha=0.5)
    for bar, val in zip(bars4, results["combined_score"]):
        ax4.text(bar.get_width() + 0.005,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=7, color="#475569")

    # zapis do pliku
    os.makedirs(output_dir, exist_ok=True)
    safe_name = dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    filepath = os.path.join(OUTPUT_DIR, f"feature_importance_{safe_name}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nZapisano wykres: {filepath}")
    return filepath


def main():
    datasets = {
        "Kupno (Sale)":    ("apartments_sale_processed.csv", "apartments_sale_mapping.json"),
        "Wynajem (Rent)":  ("apartments_rent_processed.csv", "apartments_rent_mapping.json"),
    }

    all_results = {}
    features_to_save = {}

    for name, (filename, mapping_filename) in datasets.items():
        filepath = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Brak pliku: {filepath}")
            continue

        df = pd.read_csv(filepath)
        if TARGET_COL not in df.columns:
            print(f"Brak kolumny '{TARGET_COL}' w {filename}")
            continue

        city_mapping = {}
        mapping_filepath = os.path.join(PROCESSED_DIR, mapping_filename)
        if os.path.exists(mapping_filepath):
            try:
                with open(mapping_filepath, "r", encoding="utf-8") as f:
                    full_mapping = json.load(f)
                    city_mapping = full_mapping.get("city", {})
            except Exception as e:
                print(f"Błąd wczytywania mapowania miast: {e}")

        results, k_80, k_90, cumulative = analyze_features(df, name)
        plot_results(results, k_80, k_90, cumulative, name)

        city_results = analyze_features_per_city(df, name, generate_plots=True, output_dir=OUTPUT_DIR_CITIES, city_mapping=city_mapping)
        features_to_save[name] = {}
        for c_id, c_data in city_results.items():
            k_80_val = c_data["k_80"]
            top_cols = c_data["results"]["feature"].iloc[:k_80_val].tolist()
            features_to_save[name][str(c_id)] = top_cols

        all_results[name] = {
            "results": results,
            "k_80": k_80,
            "k_90": k_90,
        }

    try:
        features_json_path = os.path.join(PROCESSED_DIR, "top_80_features.json")
        with open(features_json_path, "w", encoding="utf-8") as f:
            json.dump(features_to_save, f, indent=4, ensure_ascii=False)
        print(f"\nZapisano wyselekcjonowane cechy do pliku: {features_json_path}")
    except Exception as e:
        print(f"Błąd podczas zapisu cech do JSON: {e}")

    # Porównanie między zbiorami
    if len(all_results) == 2:
        names = list(all_results.keys())
        top_sale = set(all_results[names[0]]["results"]["feature"].iloc[:all_results[names[0]]["k_80"]])
        top_rent = set(all_results[names[1]]["results"]["feature"].iloc[:all_results[names[1]]["k_80"]])
        common = top_sale & top_rent

        print("\nPorównanie top cech między zbiorami (80% MI)")
        print(f"Wspólne cechy ({len(common)}): {sorted(common)}")
        print(f"Tylko w Kupnie: {sorted(top_sale - top_rent)}")
        print(f"Tylko w Wynajmie: {sorted(top_rent - top_sale)}")

    print("\nAnaliza zakończona!")


if __name__ == "__main__":
    main()