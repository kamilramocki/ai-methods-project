# 1. Wstępny opis zbioru danych (Raport)

## Opis danych wejściowych
Projekt wykorzystuje historyczne dane z rynku nieruchomości w Polsce (podział na transakcje kupna oraz wynajmu) z lat 2023–2024. W celu zapewnienia stabilności modeli regresyjnych i eliminacji szumu informacyjnego, z pierwotnego zbioru danych wyselekcjonowano kluczowe cechy o najwyższym stopniu korelacji z ceną końcową:

* **squareMeters** – metraż nieruchomości (zmienna ciągła)
* **rooms** – liczba pokoi (zmienna dyskretna)
* **floor** – piętro, na którym znajduje się lokal (zmienna dyskretna)
* **buildYear** – rok budowy obiektu (zmienna dyskretna)
* **price** – cena nieruchomości / cena wynajmu (zmienna docelowa ciągła)
* **city** – miasto (używana do segmentacji rynków)

Zrezygnowano z wykorzystania kolumn opisujących odległości od konkretnych punktów infrastruktury (np. szkół, przedszkoli, przystanków) w celu skupienia się na najważniejszych, bazowych cechach determinujących wartość nieruchomości i uproszczenia architektury wejściowej w początkowej fazie projektu.

## Metodologia przygotowania danych
1. **Konsolidacja**: Dane z poszczególnych miesięcy (zawarte w formacie CSV) są automatycznie wczytywane i konsolidowane w dwa główne pakiety danych: *Kupno* oraz *Wynajem*.
2. **Segmentacja i podział na rynki lokalne**: Ze względu na specyfikę rynków lokalnych (skrajnie różne średnie ceny za metr kwadratowy w różnych miastach), algorytm dokonuje segmentacji danych.
3. **Imputacja braków**: Wszystkie braki danych (np. brak roku budowy czy piętra) są automatycznie uzupełniane wartością mediany z obliczoną dla danego miasta, co zapobiega zniekształceniu wyników przez wartości skrajne (outliery) oraz utracie cennych wierszy w procesie trenowania.
4. **Wizualizacja**: Skrypt generuje histogramy rozkładu cen w sposób globalny oraz dla poszczególnych głównych miast, aby zidentyfikować kształt rozkładu i ewentualne anomalie cenowe.
5. **Strukturyzacja w pamięci**: Wyczyszczone i zgrupowane według miast dane są konwertowane na obiekty Numpy i przetrzymywane w pamięci programu (macierze oddzielone na zbiory `X` – cechy, i `y` – etykiety)

---

# 2. Pomysły na dalszy rozwój projektu

1. **Feature Engineering**: 
   * Stworzenie cech pochodnych np. `cena_za_metr` jako pomocniczy atrybut do detekcji outlierów.
   * Kategoryzacja wieku budynku (np. "stare budownictwo", "wielka płyta", "nowe budownictwo").
2. **Zaawansowana detekcja outlierów**: Zastosowanie metod takich jak Z-Score lub IQR (Interquartile Range) w celu usunięcia ekstremalnych i nierzeczywistych ofert (np. cena wynajmu 1 zł lub cena kupna 100 milionów zł), które zaburzają regresję.
3. **Implementacja modeli**: 
   * Start od prostych algorytmów regresyjnych takich jak *Regresja Liniowa*.
   * Następnie przejście do silniejszych modeli opartych na drzewach np. *Random Forest* i *XGBoost*.
4. **Hiperparametryzacja i walidacja krzyżowa**: Zastosowanie Grid Search lub Random Search oraz k-fold cross-validation dla poszczególnych miast.
