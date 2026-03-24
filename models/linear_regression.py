"""
================================================================================
EDU-ML SANDBOX — WTYCZKA MODELU
================================================================================

@model: LinearRegression
@task: regression
@name: Regresja Liniowa
@description: Najprostszy i najbardziej intuicyjny model regresji. Dopasowuje
              prostą linię do danych metodą najmniejszych kwadratów (OLS).
              Idealna do zrozumienia podstaw uczenia maszynowego — zobacz,
              jak linia "podąża" za punktami i co się dzieje, gdy dodasz outlier.
@icon: linear_regression.svg

================================================================================
PARAMETRY
================================================================================

@param: fit_intercept
@label: Dopasuj wyraz wolny (intercept)
@type: bool
@default: true
@hint: Określa czy linia może przecinać oś Y w dowolnym miejscu.
       
       WŁĄCZONE (True):
       • Linia ma postać: y = a·x + b
       • Parametr "b" (intercept) jest dopasowywany
       • Linia może zaczynać się powyżej lub poniżej zera
       • REKOMENDOWANE dla większości przypadków
       
       WYŁĄCZONE (False):
       • Linia ma postać: y = a·x
       • Linia MUSI przechodzić przez punkt (0, 0)
       • Użyj tylko gdy wiesz, że dane zaczynają się od zera
       
       EKSPERYMENT: Wyłącz i zobacz jak linia "skacze" do punktu (0,0).

@param: positive
@label: Tylko dodatnie współczynniki
@type: bool
@default: false
@hint: Wymusza, aby współczynniki (nachylenie) były nieujemne.
       
       WYŁĄCZONE (False):
       • Linia może mieć dowolne nachylenie (rosnąca lub malejąca)
       • REKOMENDOWANE dla większości przypadków
       
       WŁĄCZONE (True):
       • Linia może być tylko rosnąca lub pozioma
       • Użyj gdy wiesz, że zależność musi być dodatnia
         (np. więcej godzin nauki = wyższy wynik)
       
       EKSPERYMENT: Włącz dla danych z trendem spadkowym i zobacz efekt.

================================================================================
PARAMETRY UKRYTE (techniczne, nieistotne dla nauki)
================================================================================

@param: copy_X
@show: false

@param: n_jobs
@show: false

================================================================================
INFORMACJE DYDAKTYCZNE
================================================================================

CEL NAUKI:
Zrozumienie jak model "uczy się" z danych poprzez minimalizację błędu.

KLUCZOWE POJĘCIA:
• Współczynnik kierunkowy (slope) — jak bardzo y rośnie gdy x rośnie o 1
• Wyraz wolny (intercept) — gdzie linia przecina oś Y
• Błąd średniokwadratowy (MSE) — miara jakości dopasowania
• R² — procent wariancji wyjaśnionej przez model

PYTANIA DO EKSPERYMENTU:
1. Co się stanie gdy dodasz punkt bardzo daleko od reszty (outlier)?
2. Jak zmienia się nachylenie linii gdy wyłączysz intercept?
3. Dlaczego MSE rośnie gdy linia nie pasuje do danych?

================================================================================
"""

from sklearn.linear_model import LinearRegression

model = LinearRegression()
