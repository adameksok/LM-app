'''
================================================================================
EDU-ML SANDBOX — WTYCZKA MODELU
================================================================================

@model: KNeighborsClassifier
@task: classification
@name: K-Najbliższych Sąsiadów (KNN)
@description: Klasyfikuje nowe punkty danych na podstawie "głosowania" K najbliższych 
              punktów ze zbioru treningowego. Algorytm leniwy (lazy learning) — nie buduje
              równania podczas treningu, a jedynie zapamiętuje zbiór danych do porównań.
@icon: knn.svg

================================================================================
RÓWNANIE MODELU
================================================================================

@equation: y = M_{k}(\mathbf{x})
@cost_function: Brak klasycznej optymalizacji funkcji kosztu
@cost_name: Leniwe Uczenie (Lazy Learning)

================================================================================
PARAMETRY — WIDOCZNE DLA STUDENTA
================================================================================

@param: n_neighbors
@label: Liczba sąsiadów (K)
@type: int
@min: 1
@max: 51
@step: 2
@default: 5
@hint: Ile najbliższych punktów bierze udział w głosowaniu.
       
       MAŁE K (np. 1-3):
       • Granica decyzyjna jest bardzo "poszarpana" i dopasowuje się do każdego punktu
       • Bardzo duże ryzyko przeuczenia (overfitting)
       • Obejmuje nawet odizolowane szumy
       
       DUŻE K (np. 15-25):
       • Gładka, uogólniona granica decyzyjna
       • Odporyszna na szum (łagodzi pojedyncze kropki odstające)
       • Zbyt duże K może prowadzić do underfittingu
       
       EKSPERYMENT: Ustaw K=1 i zobacz wyspy dookoła pojedynczych punktów.

@param: weights
@label: Wagi sąsiadów
@type: select
@options: uniform, distance
@default: uniform
@hint: Jak liczyć głosy sąsiadów.
       
       UNIFORM (równe):
       • Każdy z K sąsiadów ma dokładnie taki sam 1 głos
       • Standardowe, proste zachowanie
       
       DISTANCE (ważone odległością):
       • Im punkt jest bliżej, tym jego głos jest silniejszy (waży więcej)
       • Bardzo przydatne, gdy zbiór jest niezbalansowany i małe klasy są stłoczone, 
         a klasy dominujące "zalewają" otoczenie

@param: p
@label: Metryka odległości
@type: select
@options: 1, 2
@default: 2
@hint: W jaki sposób mierzona jest przestrzeń "od jednego punktu do drugiego".
       
       1 (Manhattan / Miejska):
       • Oblicza dystans jak kroki w kratkowanych ulicach (|x1-x2| + |y1-y2|)
       • Czasem lepsze dla twardo określonych siatek cech
       
       2 (Euklidesowa / W linii prostej):
       • Klasyczna linia lotnicza od punktu do punktu
       • DOMYŚLNY WYBÓR W 99% PRZYPADKÓW

================================================================================
PARAMETRY — UKRYTE (techniczne)
================================================================================

@param: algorithm
@show: false

@param: leaf_size
@show: false

@param: metric
@show: false

@param: metric_params
@show: false

@param: n_jobs
@show: false

================================================================================
WYJŚCIA MODELU — ATRYBUTY
================================================================================

@output: classes_
@output_label: Rozpoznawane klasy
@output_type: labels
@output_show: true
@output_format: text
@output_hint: Lista klas które model rozróżnia i poddaje "głosowaniu".

@output: n_samples_fit_
@output_label: Zapamiętanych punktów bazowych
@output_type: integer
@output_show: true
@output_format: text
@output_hint: KNN nie używa współczynników ani wyrazów wolnych.
              KNN wymaga zapamiętania CAŁEGO treningowego zbioru danych 
              do działania (tzw. uczenie leniwe).
              Model wczytał do pamięci pokazaną liczbę referencyjnych punktów.

@output: effective_metric_
@output_label: Wykorzystana matematyczna metryka dystansu
@output_type: string
@output_show: true
@output_format: text
@output_hint: Metryka obliczeniowa zaszyta na zapleczu algorytmu.

================================================================================
METRYKI JAKOŚCI
================================================================================

@metric: accuracy
@metric_label: Dokładność (Accuracy)
@metric_show: true
@metric_format: percent
@metric_good_value: 0.8
@metric_hint: Procent poprawnie sklasyfikowanych próbek.
              Accuracy = (TP + TN) / (TP + TN + FP + FN)

@metric: precision
@metric_label: Precyzja (Precision)
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Z tych które model oznaczył jako pozytywne, ile naprawdę było?
              Precision = TP / (TP + FP)

@metric: recall
@metric_label: Czułość (Recall)
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Z tych które naprawdę były pozytywne, ile model znalazł?
              Recall = TP / (TP + FN)

@metric: f1
@metric_label: F1-Score
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Średnia harmoniczna Precision i Recall.
              F1 = 2 × (Precision × Recall) / (Precision + Recall)

@metric: roc_auc
@metric_label: ROC AUC
@metric_show: true
@metric_format: decimal
@metric_good_value: 0.8
@metric_hint: Pole pod krzywą ROC (Receiver Operating Characteristic).
              0.5 = rzucanie monetą, 1.0 = perfekcyjna predykcja.

================================================================================
WIZUALIZACJE
================================================================================

@visualization: decision_boundary
@viz_label: Granice Decyzyjne Sąsiedztwa
@viz_show: true
@viz_position: main

@visualization: confusion_matrix
@viz_label: Macierz pomyłek
@viz_show: true
@viz_position: side

@visualization: roc_curve
@viz_label: Krzywa ROC
@viz_show: true
@viz_position: bottom

@visualization: probability_distribution
@viz_label: Rozkład Prawdopodobieństw Głosowania
@viz_show: true
@viz_position: bottom

@visualization: precision_recall_curve
@viz_label: Krzywa Precision-Recall
@viz_show: true
@viz_position: bottom

================================================================================
'''

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
