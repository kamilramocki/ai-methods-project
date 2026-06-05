import numpy as np
import pandas as pd
import scipy.stats as stats

def main():
    results_matrix = np.load("cv_results_matrix.npy")

    datasets = ["Kupno (Sale)", "Wynajem (Rent)"]
    cities = [
        "Bialystok", "Bydgoszcz", "Czestochowa", "Gdansk", "Gdynia",
        "Katowice", "Krakow", "Lodz", "Lublin", "Poznan",
        "Radom", "Rzeszow", "Szczecin", "Warszawa", "Wroclaw"
    ]
    models = ["Linear Regression", "Random Forest", "Gradient Boosting", "KNN", "Decision Tree"]

    METRIC_R2_IDX = 3
    ALPHA_BONFERRONI = 0.05 / 10

    final_summary = []

    print("\nRozpoczynam analizę statystyczną\n")

    for d_idx, dataset_name in enumerate(datasets):
        print(f"\n Zbiór danych: {dataset_name}")

        for c_idx, city_name in enumerate(cities):
            city_r2_data = results_matrix[d_idx, c_idx, :, :, METRIC_R2_IDX]

            # globalny test friedmana
            try:
                friedman_stat, p_friedman = stats.friedmanchisquare(*[city_r2_data[m, :] for m in range(len(models))])
            except ValueError:
                p_friedman = 1.0
                friedman_stat = 0.0

            fold_ranks = np.zeros_like(city_r2_data)
            for f in range(10):
                # aby najwyższy r2 dostał range 1, podajemy ujemne wartości
                fold_ranks[:, f] = stats.rankdata(-city_r2_data[:, f])

            mean_ranks = np.mean(fold_ranks, axis=1)
            mean_r2_scores = np.mean(city_r2_data, axis=1)

            best_model_idx = np.argmin(mean_ranks)
            best_model_name = models[best_model_idx]

            print(f"\nMiejscowość: {city_name}")
            print(f"  -> Test Friedmana: p-value = {p_friedman:.6f} " + (
                "(ISTOTNY)" if p_friedman < 0.05 else "(NIEISTOTNY)"))
            print("  -> Średnie rangi i wyniki R2:")
            for m_idx, m_name in enumerate(models):
                print(f"    - {m_name:<20} | Średnia ranga: {mean_ranks[m_idx]:.2f} | Średni R2: {mean_r2_scores[m_idx]:.4f}")

            significant_winners = []
            stat_equivalents = []

            if p_friedman < 0.05:
                for m_idx, m_name in enumerate(models):
                    if m_idx == best_model_idx:
                        continue

                    p_wilcoxon = stats.wilcoxon(city_r2_data[best_model_idx, :], city_r2_data[m_idx, :])

                    if p_wilcoxon.pvalue < ALPHA_BONFERRONI:
                        significant_winners.append(m_name)
                    else:
                        stat_equivalents.append(m_name)

                if len(significant_winners) == len(models) - 1:
                    verdict = f"Model {best_model_name} jest istotnym statystycznie liderem."
                else:
                    verdict = f"Model {best_model_name} działa najlepiej, ale NIE JEST istotnie lepszy od: {', '.join(stat_equivalents)}."
            else:
                verdict = "Brak podstaw do odrzucenia hipotezy zerowej Friedmana. Wszystkie modele działają statystycznie tak samo."

            print(f"\nWERDYKT: {verdict}")

            final_summary.append({
                "Zbiór": dataset_name,
                "Miejscowość": city_name,
                "Najlepszy Regresor": best_model_name,
                "Średni R2 Lidera": mean_r2_scores[best_model_idx],
                "Średnia Ranga Lidera": mean_ranks[best_model_idx],
                "Statystycznie równorzędne modele": ", ".join(stat_equivalents) if stat_equivalents else "Brak (Lider dominuje samodzielnie)",
                "P-value Friedmana": p_friedman
            })
    summary_df = pd.DataFrame(final_summary)
    summary_df.to_csv("statistical_report.csv", index=False, encoding="utf-8")
    print("Raport końcowy 'statistical_report.csv' został wygenerowany")


if __name__ == "__main__":
    main()