# PRD — Asystent AI (Gemini 2.0 Flash) dla Edu-ML Sandbox

**Wersja:** 1.1
**Data:** 2026-03-27
**Status:** Draft
**Autor:** AI-assisted (Claude Sonnet 4.6)

---

## Spis treści

1. [Kontekst i Problem](#1-kontekst-i-problem)
2. [Cele i Zakres](#2-cele-i-zakres)
3. [Użytkownicy i User Stories](#3-użytkownicy-i-user-stories)
4. [Wymagania Funkcjonalne](#4-wymagania-funkcjonalne)
5. [Wymagania Techniczne](#5-wymagania-techniczne)
6. [Dane przekazywane do Gemini](#6-dane-przekazywane-do-gemini)
7. [Makieta UI](#7-makieta-ui)
8. [Plan Implementacji](#8-plan-implementacji)
9. [Metryki sukcesu](#9-metryki-sukcesu)
10. [Otwarte pytania](#10-otwarte-pytania--decyzje-do-podjęcia)
11. [Załączniki](#11-załączniki)

---

## 1. Kontekst i Problem

### 1.1 Stan aplikacji

Edu-ML Sandbox to edukacyjna platforma Streamlit do nauki algorytmów ML (sklearn).

**Architektura:**
- **Plugin-based:** modele opisane tagami `@tag` w docstringu Python, odkrywane przez `core/plugin_engine.py`
- **UI kompozytowy:** komponenty w `components/` (dashboard, sidebar, visualization, data_prep_ui, saved_model_view)
- **i18n PL/EN:** tłumaczenia w `translations/pl.json` i `translations/en.json`, funkcja `t()` z dot-notation, język sterowany przez `st.session_state["lang"]`
- **Session state:** centralne zarządzanie stanem w `core/session_state.py`

**Aktualnie dostępne wtyczki (modele):**

| Wtyczka | Plik | Task | Metryki |
|---------|------|------|---------|
| Linear Regression | `models/linear_regression_en.py` | regression | R², MSE, RMSE, MAE |
| Logistic Regression | `models/logistic_regression.py` | classification | accuracy, precision, recall, F1, ROC-AUC, log_loss |
| K-Nearest Neighbors | `models/knn_wtyczka.py` | classification | accuracy, precision, recall, F1, ROC-AUC |

**Kluczowy stan sesji po uruchomieniu modelu:**

```
st.session_state["model_ran"]          # bool
st.session_state["last_run_X"]         # np.ndarray
st.session_state["last_run_y"]         # np.ndarray
st.session_state["last_run_params"]    # dict {param_name: value}
st.session_state["last_run_features"]  # list[str]
st.session_state["last_run_target"]    # str
st.session_state["lang"]               # "pl" | "en"
```

### 1.2 Problem do rozwiązania

Student po uruchomieniu modelu widzi metryki i wykresy, ale **nie ma możliwości interaktywnego pytania** — „co to oznacza?", „dlaczego R²=95%?", „co to macierz pomyłek?", „dlaczego precision różni się od recall?". Musi szukać poza aplikacją, tracąc kontekst eksperymentu.

Studenci na poziomie beginning/intermediate nie potrafią samodzielnie zinterpretować wyników modelu ML. Potrzebują **kontekstowego tutora**, który:
- widzi te same wyniki co student (metryki, parametry, typ zadania)
- odpowiada w języku wybranym przez użytkownika (PL lub EN)
- używa przykładów z branży logistycznej dostosowanych do typu modelu

### 1.3 Ryzyko główne (zidentyfikowane przed implementacją)

> Bez guardrails model językowy odpowie na każde pytanie — zmienia asystenta edukacyjnego ML w generyczny chatbot, niespójny z celem aplikacji.

**Rozwiązanie:** Wielowarstwowy system walidacji tematycznej (szczegóły: FR-3).

---

## 2. Cele i Zakres

### 2.1 Cele (In Scope)

| # | Cel | Miernik sukcesu |
|---|-----|-----------------|
| G1 | Asystent wyjaśnia wyniki w języku wybranym przez użytkownika (PL/EN) | Język odpowiedzi = `st.session_state["lang"]` w 100% przypadków |
| G2 | Asystent rozumie wszystkie trzy wtyczki (regression + classification) | Poprawny kontekst dla LR, Logistic Reg., KNN |
| G3 | Odpowiedzi tylko z domeny ML | <2% błędnych odrzuceń pytań ML |
| G4 | Koszt API proporcjonalny do użycia | Lokalna whitelist jako first-pass bez kosztu API |
| G5 | Integracja nie łamie istniejącej architektury | Zero zmian w `core/plugin_engine.py` i `core/plugin_parser.py` |

### 2.2 Poza zakresem (Out of Scope — v1)

- Asystent dla widoku Dashboard (tylko experiment view)
- Tryb offline / lokalny LLM
- Historia rozmów między sesjami (persistence po zamknięciu przeglądarki)
- Głosowy interfejs
- Generowanie lub modyfikowanie kodu pluginów przez AI
- Asystent dla widoków: `saved_model_view`, `data_prep_ui`

---

## 3. Użytkownicy i User Stories

**Persona: Student logistyki, 2. rok studiów, uczy się Python i ML**

| ID | Jako... | Chcę... | Żeby... |
|----|---------|---------|---------|
| US-1 | student (PL) | zapytać „co oznacza R²=95,4%?" | zrozumieć jakość modelu bez szukania w Google |
| US-2 | student (EN) | ask "what does precision mean?" | understand the metric in English |
| US-3 | student | zapytać „dlaczego KNN z K=1 jest niestabilny?" | zrozumieć wpływ hiperparametru przed eksperymentem |
| US-4 | student | zapytać „co mi mówi macierz pomyłek?" | zinterpretować wyniki klasyfikacji (LR, KNN) |
| US-5 | student | otrzymać odpowiedź z przykładem z logistyki | powiązać teorię ML z pracą zawodową |
| US-6 | student | widzieć historię rozmowy w sesji | budować rozumowanie krok po kroku |
| US-7 | nauczyciel | mieć pewność, że asystent nie odpowiada poza ML | korzystać z aplikacji na zajęciach bez ryzyka |

---

## 4. Wymagania Funkcjonalne

### FR-1: Chat konwersacyjny

- Pole tekstowe `st.chat_input` z placeholderem (i18n: `assistant.input_placeholder`)
- Dymki wiadomości przez `st.chat_message("user")` i `st.chat_message("assistant")`
- Pełna historia rozmowy w sesji — `st.session_state["chat_history"]` jako `List[dict]`
- Format wpisu historii: `{"role": "user"|"assistant", "content": str}`
- Przycisk „Wyczyść historię" (ikona kosza)
- Maksymalnie 20 wiadomości w historii (rolling window — najstarsze odpadają)
- Spinner `st.spinner(t("assistant.thinking"))` podczas oczekiwania na odpowiedź

### FR-2: Kontekst modelu w prompcie — wszystkie wtyczki

Kontekst budowany dynamicznie przez `build_model_context()`. Sekcja **Metryki** i **Zadanie** są dostosowane do typu zadania aktywnej wtyczki.

**Przykład — regression (Linear Regression):**

```
=== EXPERIMENT CONTEXT ===
Model: Linear Regression (LinearRegression)
Task: regression
Status: trained

Configuration parameters:
  fit_intercept: True
  positive: False

Input data:
  Samples: 150
  Features: X1

Model metrics:
  R²: 95.4%
  MSE: 9.5153
  RMSE: 3.0847
  MAE: 2.4411

Learned equation: ŷ = 4.741 · X1 + 0.589
=== END OF CONTEXT ===
```

**Przykład — classification (Logistic Regression):**

```
=== EXPERIMENT CONTEXT ===
Model: Logistic Regression (LogisticRegression)
Task: classification
Status: trained

Configuration parameters:
  C: 1.0
  penalty: l2
  max_iter: 100
  class_weight: None

Input data:
  Samples: 200
  Features: distance_km, weight_kg
  Classes: [0, 1]

Model metrics:
  Accuracy: 87.5%
  Precision: 0.884
  Recall: 0.861
  F1: 0.872
  ROC-AUC: 0.921
  Log Loss: 0.312

Learned equation: σ(2.14·distance_km + 0.87·weight_kg − 1.05)
=== END OF CONTEXT ===
```

**Przykład — classification (KNN):**

```
=== EXPERIMENT CONTEXT ===
Model: K-Nearest Neighbors (KNeighborsClassifier)
Task: classification
Status: trained

Configuration parameters:
  n_neighbors: 5
  weights: uniform
  p: 2 (Euclidean)

Input data:
  Samples: 200
  Features: distance_km, weight_kg
  Classes: [0, 1]
  Stored reference points: 200

Model metrics:
  Accuracy: 83.0%
  Precision: 0.841
  Recall: 0.812
  F1: 0.826
  ROC-AUC: 0.894
=== END OF CONTEXT ===
```

**Reguły budowania kontekstu:**
- Sekcja `Learned equation` pojawia się tylko gdy `meta.task == "regression"` i model posiada `coef_` / `intercept_`
- Sekcja `Classes` pojawia się tylko gdy `meta.task == "classification"`
- Jeśli `model_ran=False`: sekcje Metryki, Równanie i Klasy są zastąpione przez `"N/A — model not trained"`
- Surowe dane (`X`, `y` arrays) **nie są przekazywane** do Gemini — tylko metryki agregowane
- Kontekst przebudowywany przy każdym pytaniu (zawsze aktualny)

### FR-3: Wielowarstwowy system walidacji tematycznej

```
Pytanie użytkownika
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  WARSTWA 1: Lokalna whitelist (0 kosztu API)             │
│                                                          │
│  Słowa kluczowe ML (PL + EN) — patrz Załącznik A        │
│  ~100 terminów: metryki, modele, koncepty, logistyka    │
│                                                          │
│  HIT → przepuść do Gemini                               │
│  MISS → Warstwa 2                                        │
└──────────────────────────────────────────────────────────┘
        │ (tylko dla niejasnych przypadków)
        ▼
┌──────────────────────────────────────────────────────────┐
│  WARSTWA 2: Walidacja LLM (dodatkowy call Gemini)        │
│                                                          │
│  Prompt: "Is the following question related to machine   │
│  learning, statistics or data analysis?                  │
│  Respond ONLY: {\"ml_related\": true/false}              │
│  Question: {question}"                                   │
│                                                          │
│  true → przepuść do głównego call                        │
│  false → zwróć odmowę (FR-3a)                           │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  WARSTWA 3: System Prompt (zawsze aktywny)               │
│                                                          │
│  Instrukcja odmowy dla off-topic wbudowana               │
│  bezpośrednio w system prompt (FR-4)                     │
└──────────────────────────────────────────────────────────┘
```

**FR-3a: Komunikat odmowy — dostosowany do języka**

Polski (`lang=pl`):
```
Mogę odpowiadać tylko na pytania dotyczące uczenia maszynowego
i wyników Twojego modelu. Spróbuj zapytać np.:
• Co oznacza R² w moim modelu?
• Dlaczego MAE różni się od RMSE?
• Co mi mówi macierz pomyłek?
• Jak wpływa parametr K w KNN?
```

English (`lang=en`):
```
I can only answer questions about machine learning and your
model results. Try asking e.g.:
• What does R² mean in my model?
• Why does MAE differ from RMSE?
• How do I read the confusion matrix?
• How does parameter K affect KNN?
```

### FR-4: System Prompt — dynamiczny, dwujęzyczny

System prompt jest **budowany dynamicznie** przy każdym wywołaniu. Język instrukcji oraz przykładów dostosowuje się do `st.session_state["lang"]`.

**Wersja polska (`lang=pl`):**

```
Jesteś asystentem edukacyjnym pomagającym studentom zrozumieć
uczenie maszynowe.

BEZWZGLĘDNE REGUŁY:
1. Odpowiadasz WYŁĄCZNIE na pytania z zakresu ML, statystyki
   i analizy danych. Jeśli pytanie dotyczy innego tematu —
   grzecznie odmów i zaproponuj pytanie ML.
2. Język odpowiedzi: POLSKI.
3. Styl: prosty, zrozumiały dla studenta 2. roku studiów.
4. Wyjaśniaj żargon ML prostym językiem na pierwszym użyciu.
5. Używaj przykładów z logistyki odpowiednich do zadania modelu:
   - regression → prognozowanie czasu dostawy, kosztów transportu
   - classification → klasyfikacja przesyłek (opóźniona/na czas),
     wykrywanie uszkodzeń paczek
6. Gdy student pyta o konkretne liczby — używaj DOKŁADNIE wartości
   z sekcji KONTEKST poniżej.
7. Odpowiedzi zwięzłe: 3–6 zdań + opcjonalny przykład.

KONTEKST AKTUALNEGO EKSPERYMENTU:
{model_context}
```

**Wersja angielska (`lang=en`):**

```
You are an educational assistant helping students understand
machine learning.

STRICT RULES:
1. Answer ONLY questions about ML, statistics and data analysis.
   If the question is off-topic — politely decline and suggest
   an ML question.
2. Response language: ENGLISH.
3. Style: simple, suitable for a 2nd-year university student.
4. Explain ML jargon in plain language on first use.
5. Use logistics examples appropriate to the model task:
   - regression → predicting delivery time, transport costs
   - classification → classifying parcels (delayed/on time),
     detecting damaged packages
6. When the student asks about specific numbers — use EXACTLY
   the values from the CONTEXT section below.
7. Keep answers concise: 3–6 sentences + optional example.

CURRENT EXPERIMENT CONTEXT:
{model_context}
```

### FR-5: Umiejscowienie w UI

**Decyzja: Nowa zakładka `💬 Asystent` / `💬 Assistant` — czwarta zakładka w experiment view**

Uzasadnienie:

| Opcja | Zalety | Wady | Ocena |
|-------|--------|------|-------|
| Nowa zakładka | Spójne z arch., pełna przestrzeń, łatwe przełączanie | Student musi zmienić zakładkę | ✅ Wybrano |
| Pod Quality Metrics | Blisko wyników | Przepełnia widok | ❌ |
| Sidebar | Zawsze widoczny | Ciasno, sidebar już zajęty nawigacją | ❌ |

**Integracja w `app.py` — `render_experiment_view()`:**

```python
tab_setup, tab_history, tab_settings, tab_assistant = st.tabs([
    f"⚡ {t('experiment.tab_setup')}",
    f"📜 {t('experiment.tab_history')}",
    f"⚙️ {t('experiment.tab_settings')}",
    f"💬 {t('experiment.tab_assistant')}",
])

with tab_assistant:
    from components.ai_chat_ui import render_ai_chat_tab
    render_ai_chat_tab(config)
```

Etykieta zakładki pochodzi z i18n: `experiment.tab_assistant` → `"💬 Asystent"` (PL) / `"💬 Assistant"` (EN).

### FR-6: Zachowanie przy zmianie modelu i języka

**Zmiana modelu** (`current_model_id` się zmienia):
- Historia czatu jest czyszczona (`chat_history = []`)
- `chat_initialized = False` — asystent wyśle nowe powitanie uwzględniające nowy model

**Zmiana języka** (`lang` się zmienia, rerun):
- Historia czatu jest czyszczona (poprzednie wiadomości mogą być w innym języku)
- Asystent wyśle nowe powitanie w nowym języku

Wykrycie obu zmian w tym samym bloku `if` co `_last_experiment_model_id` w `render_experiment_view()`.

### FR-7: Powitanie kontekstowe (auto-inicjalizacja)

Przy pierwszym wejściu na zakładkę (lub po resecie) asystent wysyła powitanie dostosowane do stanu modelu i jego typu.

Przykłady (wersja PL):

| Stan | Wtyczka | Powitanie |
|------|---------|-----------|
| Wytrenowany | Linear Regression | „Cześć! Widzę, że Twoja regresja liniowa ma R²=95,4% i równanie ŷ = 4.741·X1 + 0.589. Co chciałbyś zrozumieć?" |
| Wytrenowany | Logistic Regression | „Cześć! Twój model klasyfikacji osiągnął Accuracy=87,5% i ROC-AUC=0,921. Chętnie wyjaśnię macierz pomyłek, precision/recall lub parametr C." |
| Wytrenowany | KNN | „Cześć! Twój KNN (K=5) osiągnął Accuracy=83%. Możemy porozmawiać o wpływie K na granicę decyzyjną lub o różnicy między metrykami Manhattan i Euclidean." |
| Nie wytrenowany | (dowolny) | „Cześć! Model jeszcze nie jest wytrenowany — możesz pytać o algorytm i parametry. Gdy uruchomisz model, będę widział Twoje wyniki." |

---

## 5. Wymagania Techniczne

### TR-1: Nowe zależności

```
# requirements.txt
google-generativeai>=0.7.0
python-dotenv>=1.0.0
```

Klucz API:

```bash
# .env (już w .gitignore)
GEMINI_API_KEY=your_key_here
```

### TR-2: Nowy moduł `core/ai_assistant.py`

Czysta logika biznesowa, zero importów Streamlit — w pełni testowalny jednostkowo.

```python
# Publiczne API modułu:

ML_KEYWORDS: set[str]
# ~100 słów kluczowych ML w PL i EN (patrz Załącznik A)

SYSTEM_PROMPT_TEMPLATE_PL: str
SYSTEM_PROMPT_TEMPLATE_EN: str
# Szablony z placeholderem {model_context}

def build_model_context(session_state: dict, config: PluginConfig) -> str:
    """
    Buduje tekstowy snapshot eksperymentu z session_state.
    Sekcja Metryki i Równanie/Klasy dostosowana do config.metadata.task.
    Obsługuje: regression, classification (wszystkie obecne wtyczki).
    """

def build_welcome_message(session_state: dict, config: PluginConfig, lang: str) -> str:
    """
    Generuje powitanie kontekstowe dostosowane do wtyczki i stanu modelu.
    Nie wywołuje API — statyczny tekst z podstawionymi wartościami.
    """

def validate_question_local(question: str) -> bool:
    """Warstwa 1: sprawdza whitelist. O(n), 0 kosztu API."""

def validate_question_llm(question: str, client) -> bool:
    """Warstwa 2: jeden call Gemini, zwraca bool."""

def get_assistant_response(
    question: str,
    history: list[dict],
    model_context: str,
    lang: str,          # "pl" | "en" — wybiera właściwy system prompt
    client,
) -> str:
    """Główny call do Gemini. Raises: GeminiAPIError, GeminiQuotaError."""

def get_gemini_client(api_key: str):
    """Inicjalizuje i zwraca klienta google.generativeai."""
```

### TR-3: Nowy komponent `components/ai_chat_ui.py`

```python
def render_ai_chat_tab(config: PluginConfig) -> None:
    """
    Renderuje zakładkę asystenta AI.
    Czyta lang z st.session_state["lang"] — wszystkie komunikaty UI
    przez t() zgodnie z istniejącym systemem i18n.
    """
```

### TR-4: Rozszerzenie `core/session_state.py`

```python
# Nowe klucze w init_session_state():
"chat_history":     [],    # List[dict] — {role, content}
"chat_initialized": False, # Czy powitanie wysłane dla bieżącego modelu+języka
```

### TR-5: Rozszerzenie i18n — `translations/pl.json` i `translations/en.json`

Nowe klucze w sekcji `"experiment"` (spójne z istniejącą strukturą):

```json
// pl.json
"experiment": {
  "tab_assistant": "💬 Asystent",
  ...
}

// en.json
"experiment": {
  "tab_assistant": "💬 Assistant",
  ...
}
```

Nowa sekcja `"assistant"` w obu plikach:

| Klucz | PL | EN |
|-------|----|----|
| `assistant.input_placeholder` | Zadaj pytanie o model... | Ask a question about the model... |
| `assistant.clear_history` | 🗑 Wyczyść historię | 🗑 Clear history |
| `assistant.clear_confirm` | Historia czatu wyczyszczona. | Chat history cleared. |
| `assistant.no_api_key` | Brak klucza GEMINI_API_KEY. Dodaj do pliku .env i uruchom ponownie. | Missing GEMINI_API_KEY. Add it to .env and restart. |
| `assistant.model_not_trained_banner` | Uruchom model (▶ Run Model), aby asystent widział Twoje wyniki. | Run the model (▶ Run Model) so the assistant can see your results. |
| `assistant.thinking` | Asystent analizuje... | Assistant is thinking... |
| `assistant.error_timeout` | Przekroczono czas oczekiwania. Spróbuj ponownie. | Request timed out. Please try again. |
| `assistant.error_quota` | Osiągnięto limit API Gemini. Spróbuj za chwilę. | Gemini API quota reached. Try again shortly. |
| `assistant.error_generic` | Wystąpił błąd komunikacji z asystentem. | An error occurred communicating with the assistant. |

### TR-6: Obsługa błędów

| Scenariusz | Zachowanie w UI |
|-----------|----------------|
| Brak `GEMINI_API_KEY` | `st.warning` (i18n), `st.chat_input` zablokowany (`disabled=True`) |
| Timeout Gemini API (>10s) | `st.error` (i18n), input pozostaje aktywny |
| `ResourceExhausted` (quota) | `st.error` (i18n) |
| Model nie wytrenowany | `st.info` banner (i18n), chat aktywny z kontekstem bez metryk |
| Błąd parsowania JSON (walidacja LLM) | Fallback: przepuść pytanie do Warstwy 3 |
| Nieoczekiwany wyjątek | `st.error` (i18n) + `print(traceback)` do konsoli |

### TR-7: Parametry modelu Gemini

```python
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config={
        "temperature": 0.3,        # Niska — edukacyjna deterministyczność
        "max_output_tokens": 512,  # 3–6 zdań + przykład
        "top_p": 0.8,
    }
)
```

---

## 6. Dane przekazywane do Gemini

### 6.1 Struktura wiadomości

```python
history = [
    {"role": "user",  "parts": [system_prompt_z_kontekstem]},
    {"role": "model", "parts": ["Understood, ready to help."]},
    # ... max 18 kolejnych wiadomości z historii (rolling window) ...
    {"role": "user",  "parts": [aktualne_pytanie]},
]
```

### 6.2 Polityka prywatności danych

| Dane | Przekazywane? | Uzasadnienie |
|------|:-------------:|--------------|
| Metryki modelu (R², MSE, accuracy, F1 itd.) | ✅ | Niezbędne do kontekstu |
| Parametry modelu (fit_intercept, C, K itd.) | ✅ | Niezbędne do kontekstu |
| Nazwy cech (feature_names) | ✅ | Niezbędne do kontekstu |
| Liczba próbek, liczba klas | ✅ | Niezbędne do kontekstu |
| Nazwa i typ modelu (task) | ✅ | Niezbędne do kontekstu |
| Surowe dane X (tablica wartości) | ❌ | Prywatność + koszt tokenów |
| Surowe dane y (tablica wartości) | ❌ | Prywatność + koszt tokenów |

---

## 7. Makieta UI

### 7.1 Regression — po treningu (wersja PL)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Linear Regression                                    [SAVE MODEL]  │
├─────────────────────────────────────────────────────────────────────┤
│  ⚡ Model Setup   📜 History   ⚙️ Settings   💬 Asystent ← aktywna  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 🤖  Kontekst: Linear Regression · R²=95.4% · 150 próbek     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║ 🤖 Asystent                                                   ║  │
│  ║ Cześć! Twoja regresja liniowa ma R²=95,4% i równanie         ║  │
│  ║ ŷ = 4.741·X1 + 0.589. Co chciałbyś zrozumieć?               ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│                      ╔══════════════════════════════════════╗       │
│                      ║ 👤 Ty                                ║       │
│                      ║ Co oznacza współczynnik 4.741?       ║       │
│                      ╚══════════════════════════════════════╝       │
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║ 🤖 Asystent                                                   ║  │
│  ║ Współczynnik 4.741 to nachylenie prostej. Oznacza: gdy X1   ║  │
│  ║ wzrośnie o 1, prognoza wzrośnie o ~4.741.                    ║  │
│  ║                                                               ║  │
│  ║ Przykład z logistyki: jeśli X1 to odległość (×100 km),       ║  │
│  ║ model przewiduje, że każde 100 km dodaje ~4,7 h do dostawy.  ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  [ Zadaj pytanie o model...                     ]       [➤ Wyślij]  │
│                                              [🗑 Wyczyść historię]  │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Classification — po treningu (wersja EN)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Logistic Regression                                  [SAVE MODEL]  │
├─────────────────────────────────────────────────────────────────────┤
│  ⚡ Model Setup   📜 History   ⚙️ Settings   💬 Assistant ← active  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 🤖  Context: Logistic Regression · Accuracy=87.5% · C=1.0   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║ 🤖 Assistant                                                  ║  │
│  ║ Hi! Your logistic regression reached Accuracy=87.5% and      ║  │
│  ║ ROC-AUC=0.921. What would you like to understand?             ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│                    ╔════════════════════════════════════════╗       │
│                    ║ 👤 You                                  ║       │
│                    ║ Why is precision different from recall? ║       │
│                    ╚════════════════════════════════════════╝       │
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║ 🤖 Assistant                                                  ║  │
│  ║ Precision (0.884) = of all parcels predicted "delayed",       ║  │
│  ║ how many actually were. Recall (0.861) = of all truly         ║  │
│  ║ delayed parcels, how many the model caught.                   ║  │
│  ║ Your model is slightly better at avoiding false alarms        ║  │
│  ║ than at catching all real delays.                             ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  [ Ask a question about the model...            ]       [➤ Send  ]  │
│                                                  [🗑 Clear history]  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Plan Implementacji

### Faza 1 — Backend (core)

| Krok | Plik | Opis |
|------|------|------|
| 1.1 | `requirements.txt` | Dodać `google-generativeai>=0.7.0`, `python-dotenv>=1.0.0` |
| 1.2 | `core/ai_assistant.py` | Nowy moduł: `ML_KEYWORDS`, `build_model_context()` (obsługa regression + classification), `build_welcome_message()`, `validate_question_local()`, `validate_question_llm()`, `get_assistant_response()` (parametr `lang`), `get_gemini_client()` |
| 1.3 | `core/session_state.py` | Dodać klucze `chat_history` i `chat_initialized` |

### Faza 2 — Frontend (UI)

| Krok | Plik | Opis |
|------|------|------|
| 2.1 | `components/ai_chat_ui.py` | Nowy komponent: `render_ai_chat_tab()` — czyta `lang` z session_state |
| 2.2 | `app.py` | Dodać 4. tab `tab_assistant` + reset historii przy zmianie modelu/języka |
| 2.3 | `translations/pl.json` | Dodać `experiment.tab_assistant` + sekcję `assistant.*` |
| 2.4 | `translations/en.json` | Dodać `experiment.tab_assistant` + sekcję `assistant.*` |

### Faza 3 — Jakość i testy

| Krok | Opis |
|------|------|
| 3.1 | Whitelist smoke test — 20 pytań ML (PL+EN), 20 off-topic |
| 3.2 | LLM validator test — graniczne przypadki |
| 3.3 | Kontekst regression — pytanie o R²/RMSE/równanie → wartości z sesji w odpowiedzi |
| 3.4 | Kontekst classification — pytanie o precision/recall/macierz pomyłek → wartości z sesji |
| 3.5 | Dwujęzyczność — zmiana lang PL↔EN → odpowiedź w nowym języku, historia wyczyszczona |
| 3.6 | Zmiana wtyczki LR→KNN→Logistic → chat reset, powitanie dostosowane do nowego modelu |

### Diagram zależności modułów

```
app.py
  └── components/ai_chat_ui.py
        ├── core/ai_assistant.py
        │     ├── google.generativeai
        │     └── core/plugin_interface.py  (PluginConfig)
        ├── core/i18n_utils.py              (t())
        └── st.session_state               (chat_history, chat_initialized, lang)
```

---

## 9. Metryki sukcesu

| Metryka | Target | Metoda pomiaru |
|---------|--------|----------------|
| Język odpowiedzi zgodny z `lang` | 100% | Weryfikacja 20 próbek PL + 20 EN |
| Brak błędnych odrzuceń pytań ML | <2% | Test set 50 pytań ML |
| Kontekst modelu w odpowiedzi | ≥90% odpowiedzi o wynikach zawiera aktualne wartości | Manualna weryfikacja |
| Czas odpowiedzi p95 | <4 sekundy | Pomiar w przeglądarce |
| Koszt API / sesja | <$0.01 (Gemini 2.0 Flash pricing) | Dashboard GCP |
| Brak regresji w UI | Wszystkie istniejące taby i wtyczki działają poprawnie | Smoke test |

---

## 10. Otwarte pytania / Decyzje do podjęcia

| # | Pytanie | Opcje | Rekomendacja |
|---|---------|-------|-------------|
| D1 | Warstwa 2 (LLM validation) — zawsze czy na threshold? | Zawsze / Tylko gdy whitelist miss | Threshold — oszczędza ~40% calls |
| D2 | Limit historii — 20 wiadomości czy konfigurowalny? | Stały / Ustawienie w Settings | Stały v1, konfigurowalny v2 |
| D3 | Streaming odpowiedzi token-by-token? | Tak (lepszy UX) / Nie (prostsze) | Nie w v1 — `response.text` wystarczające |
| D4 | Deploy: `.env` vs Streamlit Secrets? | `.env` + dotenv / `st.secrets` | `.env` lokalnie, `st.secrets` na Streamlit Cloud |

> **Uwaga:** Język asystenta (PL/EN) **nie jest otwartą decyzją** — jest wymaganiem (FR-4). Asystent zawsze podąża za `st.session_state["lang"]`, spójnie z istniejącym systemem i18n.

---

## 11. Załączniki

### Załącznik A: ML_KEYWORDS — lista słów kluczowych (Warstwa 1)

```python
ML_KEYWORDS = {
    # Metryki — regression
    "r2", "r²", "mse", "rmse", "mae", "błąd", "error", "dopasowanie",
    # Metryki — classification
    "accuracy", "dokładność", "precision", "precyzja", "recall", "czułość",
    "f1", "auc", "roc", "log_loss", "entropia", "macierz", "confusion",
    # Modele
    "regresja", "klasyfikacja", "klaster", "regression", "classification",
    "clustering", "knn", "svm", "drzewo", "forest", "tree", "neural",
    "logistyczna", "logistic", "liniowa", "linear",
    # Koncepty ML
    "model", "trening", "training", "predykcja", "prediction", "overfitting",
    "underfitting", "bias", "wariancja", "variance", "regularyzacja",
    "regularization", "gradient", "epoch", "batch", "learning_rate",
    "parametr", "parameter", "hiperparametr", "hyperparameter",
    "współczynnik", "coefficient", "intercept", "nachylenie", "slope",
    "sigmoid", "logit", "próg", "threshold",
    # Dane
    "cecha", "feature", "próbka", "sample", "zbiór", "dataset",
    "target", "label", "etykieta", "kolumna", "column", "klasa", "class",
    # Statystyka
    "korelacja", "correlation", "rozkład", "distribution", "odchylenie",
    "deviation", "średnia", "mean", "mediana", "median", "prawdopodobieństwo",
    "probability", "normalizacja", "normalization", "standaryzacja",
    # Wizualizacje
    "wykres", "chart", "scatter", "krzywa", "curve", "granica", "boundary",
    # Logistyka (kontekst branżowy)
    "dostawa", "opóźnienie", "przesyłka", "transport", "logistyka",
    "delivery", "shipment", "delay", "paczka", "parcel", "trasa", "route",
    # KNN-specific
    "sąsiad", "neighbor", "odległość", "distance", "manhattan", "euclidean",
    "k-nearest", "głosowanie", "voting",
    # Logistic-specific
    "c", "penalty", "l1", "l2", "lasso", "ridge", "konwergencja", "convergence",
    "max_iter", "iteracja", "iteration",
}
```

### Załącznik B: Przykładowe pytania testowe (smoke test)

**Pytania ML (powinny być przepuszczone):**

| # | Pytanie | Język | Model |
|---|---------|-------|-------|
| 1 | Co oznacza R²=95,4% w moim modelu? | PL | Linear Reg. |
| 2 | Dlaczego RMSE jest większy niż MAE? | PL | Linear Reg. |
| 3 | What does the confusion matrix tell me? | EN | Logistic Reg. |
| 4 | Why is precision different from recall? | EN | Logistic Reg. |
| 5 | Co się stanie jak zwiększę K do 15? | PL | KNN |
| 6 | Jak działa odległość Manhattan vs Euclidean? | PL | KNN |
| 7 | What is regularization parameter C? | EN | Logistic Reg. |
| 8 | Jak interpretować współczynnik regresji? | PL | Linear Reg. |

**Pytania off-topic (powinny być odrzucone):**

| # | Pytanie |
|---|---------|
| 1 | Napisz mi wiersz o wiośnie |
| 2 | Jaka jest stolica Francji? |
| 3 | Write me a JavaScript function |
| 4 | Co to jest blockchain? |
| 5 | Kiedy był II rozbiór Polski? |

---

*Dokument przygotowany na podstawie analizy kodu źródłowego Edu-ML Sandbox (commit: 60a04f5).
Wersja 1.1 — uzupełniona o obsługę wszystkich wtyczek (regression + classification) i pełną dwujęzyczność.*
