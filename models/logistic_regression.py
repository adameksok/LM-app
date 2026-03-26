"""
================================================================================
EDU-ML SANDBOX — WTYCZKA MODELU
================================================================================

@model: LogisticRegression
@task: classification
@name: Regresja Logistyczna
@description: Mimo nazwy "regresja", to model KLASYFIKACYJNY. Przewiduje 
              prawdopodobieństwo przynależności do klasy. Używa funkcji 
              sigmoidalnej (logistycznej) do zamiany wyniku liniowego 
              na prawdopodobieństwo 0-100%. Świetny punkt startowy dla 
              klasyfikacji binarnej — prosty, interpretowalny, szybki.
@icon: logistic_regression.svg

================================================================================
RÓWNANIE MODELU
================================================================================

@equation: P(y=1) = 1 / (1 + e^(-(w₁x₁ + w₂x₂ + ... + b)))
@cost_function: Log Loss = -Σ[y·log(p) + (1-y)·log(1-p)]
@cost_name: Log Loss (Cross-Entropy)

================================================================================
PARAMETRY — WIDOCZNE DLA STUDENTA
================================================================================

@param: C
@label: Siła regularyzacji (C)
@type: float
@min: 0.01
@max: 100
@step: log
@default: 1.0
@hint: Odwrotność siły regularyzacji — MNIEJSZE C = SILNIEJSZA regularyzacja.
       
       MAŁE C (0.01 - 0.1):
       • Silna regularyzacja, model "ostrożny"
       • Współczynniki bliskie zeru
       • Prostsza granica decyzyjna
       • Mniejsze ryzyko przeuczenia
       • Użyj gdy masz mało danych lub dużo cech
       
       DUŻE C (10 - 100):
       • Słaba regularyzacja, model "pewny siebie"
       • Współczynniki mogą być duże
       • Bardziej złożona granica
       • Ryzyko przeuczenia
       • Użyj gdy masz dużo danych
       
       EKSPERYMENT: Zacznij od C=1, zmniejszaj jeśli model się przeuczy.

@param: penalty
@label: Typ regularyzacji
@type: select
@options: l2, l1, none
@default: l2
@hint: Metoda "karania" modelu za zbyt duże współczynniki.
       
       L2 (Ridge):
       • Kara proporcjonalna do KWADRATU współczynników
       • Zmniejsza wszystkie współczynniki, ale nie zeruje
       • DOMYŚLNA i zalecana w większości przypadków
       
       L1 (Lasso):
       • Kara proporcjonalna do WARTOŚCI BEZWZGLĘDNEJ
       • Może ZEROWAĆ niektóre współczynniki → selekcja cech!
       • Użyj gdy podejrzewasz że część cech jest nieważna
       • UWAGA: Wymaga solver='saga'
       
       none:
       • Brak regularyzacji
       • Ryzyko przeuczenia
       • Użyj tylko z dużą ilością danych

@param: max_iter
@label: Maksymalna liczba iteracji
@type: int
@min: 50
@max: 1000
@step: 50
@default: 100
@hint: Ile kroków optymalizacji model może wykonać.
       
       MAŁA WARTOŚĆ (50-100):
       • Szybsze uczenie
       • Ryzyko że model się nie ustabilizuje
       
       DUŻA WARTOŚĆ (500-1000):
       • Model ma czas na zbieżność
       • Wolniejsze uczenie
       
       💡 Jeśli widzisz ostrzeżenie "nie osiągnięto zbieżności" 
          (ConvergenceWarning), zwiększ max_iter.
       
       EKSPERYMENT: Obserwuj n_iter_ w wynikach — jeśli równa max_iter,
                    model mógł się nie ustabilizować!

@param: class_weight
@label: Wagi klas
@type: select
@options: none, balanced
@default: none
@hint: Jak traktować niezbalansowane klasy (np. 90% klasy A, 10% klasy B).
       
       none:
       • Wszystkie klasy równie ważne
       • OK gdy klasy są zbalansowane
       
       balanced:
       • Automatycznie zwiększa wagę rzadszej klasy
       • Model bardziej "stara się" poprawnie klasyfikować mniejszość
       • Użyj gdy jedna klasa jest znacznie rzadsza
       
       EKSPERYMENT: Dla niezbalansowanych danych porównaj wyniki
                    z none vs balanced — zobacz jak zmienia się Recall!

================================================================================
PARAMETRY — UKRYTE (techniczne)
================================================================================

@param: solver
@show: false

@param: tol
@show: false

@param: fit_intercept
@show: false

@param: intercept_scaling
@show: false

@param: random_state
@show: false

@param: dual
@show: false

@param: verbose
@show: false

@param: warm_start
@show: false

@param: n_jobs
@show: false

@param: l1_ratio
@show: false

@param: multi_class
@show: false

================================================================================
WYJŚCIA MODELU — ATRYBUTY
================================================================================

@output: coef_
@output_label: Współczynniki (log-odds)
@output_type: matrix
@output_show: true
@output_format: bar_chart
@output_hint: Współczynniki w skali logarytmu szans (log-odds).
              
              INTERPRETACJA:
              • Współczynnik +1 oznacza: wzrost cechy o 1 zwiększa 
                log-odds o 1, czyli szansa rośnie ~2.7× (e^1 ≈ 2.718)
              • Współczynnik -1 oznacza: szansa maleje ~2.7×
              • Współczynnik 0 oznacza: cecha nie wpływa na wynik
              
              ZNAK:
              • Dodatni (+) → cecha ZWIĘKSZA szansę klasy 1
              • Ujemny (-) → cecha ZMNIEJSZA szansę klasy 1
              
              WIELKOŚĆ:
              • |coef| > 1 → silny wpływ
              • |coef| < 0.5 → słaby wpływ

@output: intercept_
@output_label: Wyraz wolny (bias)
@output_type: vector
@output_show: true
@output_format: text
@output_hint: Bazowy log-odds gdy wszystkie cechy = 0.
              
              Jeśli intercept > 0: model domyślnie "preferuje" klasę 1
              Jeśli intercept < 0: model domyślnie "preferuje" klasę 0
              
              To "punkt startowy" przed uwzględnieniem cech.

@output: classes_
@output_label: Rozpoznawane klasy
@output_type: labels
@output_show: true
@output_format: text
@output_hint: Lista klas które model rozróżnia.
              Kolejność ma znaczenie — klasa 1 to ta "pozytywna" 
              (dla której liczymy prawdopodobieństwo).

@output: n_iter_
@output_label: Liczba wykonanych iteracji
@output_type: integer
@output_show: true
@output_format: text
@output_hint: Ile iteracji optymalizacji wykonał model.
              
              ⚠️ WAŻNE: Jeśli n_iter_ = max_iter, model mógł się 
              NIE ustabilizować! Zwiększ max_iter i uruchom ponownie.
              
              💡 Jeśli n_iter_ << max_iter, model szybko osiągnął 
              zbieżność — to dobrze!

@output: n_features_in_
@output_label: Liczba cech
@output_type: integer
@output_show: false

@output: feature_names_in_
@output_show: false

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
              
              ✅ ZALETY: Intuicyjna, łatwa do zrozumienia
              ⚠️ WADY: Myląca gdy klasy niezbalansowane!
              
              Przykład problemu: 95% klasy A, 5% klasy B
              Model mówiący zawsze "A" ma 95% accuracy, ale jest bezużyteczny.

@metric: precision
@metric_label: Precyzja (Precision)
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Z tych które model oznaczył jako pozytywne, ile naprawdę było?
              
              Precision = TP / (TP + FP)
              
              "Jak bardzo mogę ZAUFAĆ pozytywnej predykcji?"
              
              WAŻNA GDY: Fałszywy alarm jest kosztowny
              • Filtr spamu: email oznaczony jako spam trafia do kosza
              • Diagnoza: fałszywie pozytywna = niepotrzebny stres
              
              Wysoka Precision = mało fałszywych alarmów

@metric: recall
@metric_label: Czułość (Recall)
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Z tych które naprawdę były pozytywne, ile model znalazł?
              
              Recall = TP / (TP + FN)
              
              "Jak dobrze model ZNAJDUJE pozytywne przypadki?"
              
              WAŻNA GDY: Przeoczenie jest kosztowne
              • Wykrywanie choroby: przeoczenie = brak leczenia
              • Wykrywanie fraudów: przeoczenie = strata pieniędzy
              
              Wysoki Recall = mało przeoczeń

@metric: f1
@metric_label: F1-Score
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Średnia harmoniczna Precision i Recall.
              
              F1 = 2 × (Precision × Recall) / (Precision + Recall)
              
              DLACZEGO HARMONICZNA?
              • Karze ekstremalne wartości
              • F1=0.9 wymaga OBIE metryki wysokie
              • Jeśli jedna = 0, F1 = 0
              
              UŻYJ GDY: Zależy Ci na równowadze między 
              fałszywymi alarmami a przeoczeniami.

@metric: roc_auc
@metric_label: ROC AUC
@metric_show: true
@metric_format: decimal
@metric_good_value: 0.8
@metric_hint: Pole pod krzywą ROC (Receiver Operating Characteristic).
              
              INTERPRETACJA:
              • 0.5 = losowe zgadywanie (bezużyteczny model)
              • 0.7-0.8 = akceptowalny model
              • 0.8-0.9 = dobry model
              • 0.9+ = świetny model
              • 1.0 = idealne rozróżnienie
              
              ZALETA: Nie zależy od progu decyzyjnego!
              Mierzy jak dobrze model RANKUJE przykłady
              (czy pozytywne mają wyższe prawdopodobieństwa niż negatywne).

@metric: log_loss
@metric_label: Log Loss
@metric_show: false
@metric_format: decimal
@metric_hint: Funkcja kosztu używana podczas treningu.
              Im mniejsza, tym lepiej model przewiduje prawdopodobieństwa.

================================================================================
WIZUALIZACJE
================================================================================

@visualization: equation
@viz_label: Równanie logitowe
@viz_show: true
@viz_position: top

@visualization: decision_boundary
@viz_label: Granice 
@viz_show: true
@viz_position: main

@visualization: coefficients_bar
@viz_label: Wpływ cech (wykres słupkowy)
@viz_show: true
@viz_position: side

@visualization: coefficients_table
@viz_label: Tabela współczynników
@viz_show: true
@viz_position: side

@visualization: confusion_matrix
@viz_label: Macierz pomyłek
@viz_show: true
@viz_position: side

@visualization: roc_curve
@viz_label: Krzywa ROC
@viz_show: true
@viz_position: bottom

@visualization: probability_distribution
@viz_label: Rozkład prawdopodobieństw
@viz_show: true
@viz_position: bottom

@visualization: precision_recall_curve
@viz_label: Krzywa Precision-Recall
@viz_show: true
@viz_position: bottom

================================================================================
INFORMACJE DYDAKTYCZNE
================================================================================

CEL NAUKI:
Zrozumienie klasyfikacji probabilistycznej i funkcji logistycznej.

KLUCZOWE POJĘCIA:
• Funkcja sigmoidalna: σ(z) = 1/(1+e^(-z)) — zamienia dowolną liczbę na 0-1
• Log-odds (logit): log(p/(1-p)) — logarytm szans
• Prawdopodobieństwo vs. predykcja: model daje P, próg zamienia na klasę
• Regularyzacja: karanie dużych współczynników

RÓŻNICA OD REGRESJI LINIOWEJ:
• Regresja liniowa: przewiduje LICZBĘ (cena, wiek)
• Regresja logistyczna: przewiduje PRAWDOPODOBIEŃSTWO (0-100%)

PYTANIA DO EKSPERYMENTU:
1. Jak zmienia się granica decyzyjna gdy zmniejszasz C?
2. Co się dzieje gdy włączysz regularyzację L1? Które współczynniki znikają?
3. Jak class_weight="balanced" wpływa na Recall mniejszościowej klasy?
4. Kiedy n_iter_ = max_iter? Co to oznacza?

================================================================================
"""

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
