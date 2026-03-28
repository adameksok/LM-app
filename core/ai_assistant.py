"""
core/ai_assistant.py
AI assistant backend — pure logic, no Streamlit imports.
Uses google-genai SDK (google.genai).
"""

import json
import math
from google import genai
from google.genai import types
from core.plugin_interface import PluginConfig

# ---------------------------------------------------------------------------
# ML keyword whitelist (Layer 1 validation)
# ---------------------------------------------------------------------------

ML_KEYWORDS = {
    # Regression metrics
    "r2", "r²", "mse", "rmse", "mae", "błąd", "blad", "error", "dopasowanie",
    # Classification metrics
    "accuracy", "dokładność", "dokladnosc", "precision", "precyzja",
    "recall", "czułość", "czulosc", "f1", "auc", "roc", "log_loss",
    "entropia", "macierz", "confusion", "matrix", "krzywa", "curve",
    # Model names / tasks
    "regresja", "klasyfikacja", "klaster", "regression", "classification",
    "clustering", "knn", "svm", "drzewo", "forest", "tree", "neural",
    "logistyczna", "logistic", "liniowa", "linear", "neighbors", "sąsiad", "sasiad",
    # Core ML concepts
    "model", "trening", "training", "predykcja", "prediction", "overfitting",
    "underfitting", "bias", "wariancja", "variance", "regularyzacja",
    "regularization", "gradient", "epoch", "batch", "learning_rate",
    "parametr", "parameter", "hiperparametr", "hyperparameter",
    "współczynnik", "wspolczynnik", "coefficient", "intercept",
    "nachylenie", "slope", "sigmoid", "logit", "próg", "prog", "threshold",
    # Data concepts
    "cecha", "feature", "próbka", "probka", "sample", "zbiór", "zbior",
    "dataset", "target", "label", "etykieta", "kolumna", "column",
    "klasa", "class",
    # Statistics
    "korelacja", "correlation", "rozkład", "rozklad", "distribution",
    "odchylenie", "deviation", "średnia", "srednia", "mean", "mediana",
    "median", "prawdopodobieństwo", "prawdopodobienstwo", "probability",
    "normalizacja", "normalization", "standaryzacja",
    # Visualizations
    "wykres", "chart", "scatter", "granica", "boundary", "decision",
    # Logistics domain context
    "dostawa", "opóźnienie", "opoznienie", "przesyłka", "przesylka",
    "transport", "logistyka", "delivery", "shipment", "delay",
    "paczka", "parcel", "trasa", "route",
    # KNN specific
    "odległość", "odleglosc", "distance", "manhattan", "euclidean",
    "głosowanie", "glosowanie", "voting",
    # Logistic regression specific
    "penalty", "l1", "l2", "lasso", "ridge", "konwergencja", "convergence",
    "max_iter", "iteracja", "iteration",
    # Common question words in ML context (specific enough to not false-positive)
    "wyjaśnij", "wyjasnij", "oznacza", "interpret", "explain", "means",
    "wpływ", "wplyw", "overfitting", "underfitting",
}

# ---------------------------------------------------------------------------
# System prompt templates (PL / EN)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE_PL = """Jesteś asystentem edukacyjnym pomagającym polskim studentom zrozumieć uczenie maszynowe.

BEZWZGLĘDNE REGUŁY:
1. Odpowiadasz WYŁĄCZNIE na pytania z zakresu ML, statystyki i analizy danych. Inne tematy — grzecznie odmów.
2. Język odpowiedzi: POLSKI.
3. Gdy pytanie dotyczy konkretnego modelu studenta (np. "czy mój model ma bias?", "co oznacza moje R²?") — NAJPIERW odpowiedz bezpośrednio na podstawie KONTEKSTU EKSPERYMENTU, a dopiero potem (jeśli potrzeba) krótko wyjaśnij pojęcie.
4. Żargon ML tłumacz tylko gdy pytanie jest ogólne ("co to jest bias?") lub student wyraźnie prosi o wyjaśnienie. Nie zaczynaj każdej odpowiedzi od definicji.
5. Używaj przykładów z logistyki gdy są pomocne:
   - regression → prognozowanie czasu dostawy, kosztów transportu
   - classification → klasyfikacja przesyłek (opóźniona/na czas)
6. Używaj DOKŁADNIE wartości liczbowych z sekcji KONTEKST EKSPERYMENTU.
7. Odpowiedzi zwięzłe: 3-6 zdań.

KONTEKST AKTUALNEGO EKSPERYMENTU:
{model_context}"""

SYSTEM_PROMPT_TEMPLATE_EN = """You are an educational assistant helping students understand machine learning.

STRICT RULES:
1. Answer ONLY questions about ML, statistics and data analysis. Off-topic — politely decline.
2. Response language: ENGLISH.
3. When the question is about the student's specific model (e.g. "does my model have bias?", "what does my R² mean?") — FIRST give a direct answer based on the EXPERIMENT CONTEXT, then briefly explain the concept only if needed.
4. Explain ML jargon only when the question is general ("what is bias?") or the student explicitly asks. Do NOT start every answer with a definition.
5. Use logistics examples when helpful:
   - regression → predicting delivery time, transport costs
   - classification → classifying parcels (delayed/on time)
6. Use EXACTLY the numeric values from the EXPERIMENT CONTEXT section.
7. Keep answers concise: 3-6 sentences.

CURRENT EXPERIMENT CONTEXT:
{model_context}"""


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def get_gemini_client(api_key: str) -> genai.Client:
    """Initializes and returns a google-genai Client."""
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_model_context(session_state: dict, config: PluginConfig) -> str:
    """
    Builds a structured text snapshot of the current experiment.
    Metric section adapts to task type:
      - regression    → R², MSE, RMSE, MAE, learned equation
      - classification → accuracy, precision, recall, F1, ROC-AUC, classes
    Raw X/y arrays are NOT included (privacy + token cost).
    """
    meta = config.metadata
    model_ran = session_state.get("model_ran", False)
    last_params = session_state.get("last_run_params", {})
    feature_names = session_state.get("last_run_features", [])
    run_X = session_state.get("last_run_X")
    run_y = session_state.get("last_run_y")

    lines = ["=== EXPERIMENT CONTEXT ==="]
    lines.append(f"Model: {meta.name} ({meta.model_class})")
    lines.append(f"Task: {meta.task}")
    lines.append(f"Status: {'trained' if model_ran else 'not trained'}")
    lines.append("")

    # --- Parameters ---
    if last_params:
        lines.append("Configuration parameters:")
        for k, v in last_params.items():
            lines.append(f"  {k}: {v}")
    elif config.parameters:
        lines.append("Configuration parameters (defaults):")
        for p in config.parameters:
            if p.show:
                lines.append(f"  {p.name}: {p.default}")
    lines.append("")

    # --- Data info ---
    if run_X is not None:
        lines.append("Input data:")
        lines.append(f"  Samples: {run_X.shape[0]}")
        feat_str = ", ".join(feature_names) if feature_names else f"{run_X.shape[1]} feature(s)"
        lines.append(f"  Features: {feat_str}")
        if model_ran and meta.task == "classification" and run_y is not None:
            classes = sorted(list(set(run_y.tolist())))
            lines.append(f"  Classes: {classes}")
    else:
        lines.append("Input data: N/A — no data loaded")
    lines.append("")

    # --- Metrics ---
    if model_ran and run_X is not None and run_y is not None:
        model_inst = config.model_instance
        try:
            y_pred = model_inst.predict(run_X)

            if meta.task == "regression":
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                r2 = r2_score(run_y, y_pred)
                mse = mean_squared_error(run_y, y_pred)
                rmse = math.sqrt(mse)
                mae = mean_absolute_error(run_y, y_pred)
                lines.append("Model metrics:")
                lines.append(f"  R2: {r2:.4f} ({r2 * 100:.1f}%)")
                lines.append(f"  MSE: {mse:.4f}")
                lines.append(f"  RMSE: {rmse:.4f}")
                lines.append(f"  MAE: {mae:.4f}")
                lines.append("")
                if hasattr(model_inst, "coef_") and hasattr(model_inst, "intercept_"):
                    coefs = model_inst.coef_.flatten()
                    intercept = float(
                        model_inst.intercept_[0]
                        if hasattr(model_inst.intercept_, "__len__")
                        else model_inst.intercept_
                    )
                    feat_labels = feature_names or [f"X{i+1}" for i in range(len(coefs))]
                    if len(coefs) == 1:
                        lines.append(f"Learned equation: y = {coefs[0]:.4f} * {feat_labels[0]} + {intercept:.4f}")
                    else:
                        terms = " + ".join(f"{c:.4f}*{n}" for c, n in zip(coefs, feat_labels))
                        lines.append(f"Learned equation: y = {terms} + {intercept:.4f}")

            elif meta.task == "classification":
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                )
                n_classes = len(set(run_y.tolist()))
                avg = "binary" if n_classes == 2 else "weighted"
                acc = accuracy_score(run_y, y_pred)
                prec = precision_score(run_y, y_pred, average=avg, zero_division=0)
                rec = recall_score(run_y, y_pred, average=avg, zero_division=0)
                f1 = f1_score(run_y, y_pred, average=avg, zero_division=0)
                lines.append("Model metrics:")
                lines.append(f"  Accuracy: {acc:.4f} ({acc * 100:.1f}%)")
                lines.append(f"  Precision: {prec:.4f}")
                lines.append(f"  Recall: {rec:.4f}")
                lines.append(f"  F1: {f1:.4f}")
                try:
                    if avg == "binary":
                        y_prob = model_inst.predict_proba(run_X)[:, 1]
                        auc = roc_auc_score(run_y, y_prob)
                    else:
                        y_prob = model_inst.predict_proba(run_X)
                        auc = roc_auc_score(run_y, y_prob, multi_class="ovr")
                    lines.append(f"  ROC-AUC: {auc:.4f}")
                except Exception:
                    pass
                lines.append("")
                if hasattr(model_inst, "coef_") and hasattr(model_inst, "intercept_"):
                    coefs = model_inst.coef_.flatten()
                    feat_labels = feature_names or [f"X{i+1}" for i in range(len(coefs))]
                    intercept = float(
                        model_inst.intercept_[0]
                        if hasattr(model_inst.intercept_, "__len__")
                        else model_inst.intercept_
                    )
                    terms = " + ".join(f"{c:.4f}*{n}" for c, n in zip(coefs, feat_labels))
                    lines.append(f"Learned equation: sigmoid({terms} + {intercept:.4f})")

        except Exception as e:
            lines.append(f"Metrics: unavailable ({e})")
    else:
        lines.append("Metrics: N/A — model not trained")
        lines.append("Learned equation: N/A — model not trained")

    lines.append("=== END OF CONTEXT ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Welcome message (no API call)
# ---------------------------------------------------------------------------

def build_welcome_message(session_state: dict, config: PluginConfig, lang: str) -> str:
    """
    Returns a contextual welcome string. No API call.
    Adapts to plugin type (regression/classification) and trained state.
    """
    meta = config.metadata
    model_ran = session_state.get("model_ran", False)
    run_X = session_state.get("last_run_X")
    run_y = session_state.get("last_run_y")
    last_params = session_state.get("last_run_params", {})

    if not model_ran or run_X is None or run_y is None:
        if lang == "pl":
            return (
                f"Cześć! Model **{meta.name}** nie jest jeszcze wytrenowany — "
                f"możesz pytać o algorytm i parametry. "
                f"Gdy uruchomisz model, będę widział Twoje wyniki i metryki."
            )
        return (
            f"Hi! **{meta.name}** hasn't been trained yet — "
            f"you can ask about the algorithm and its parameters. "
            f"Once you run the model, I'll be able to see your results and metrics."
        )

    try:
        model_inst = config.model_instance
        y_pred = model_inst.predict(run_X)

        if meta.task == "regression":
            from sklearn.metrics import r2_score
            r2 = r2_score(run_y, y_pred)
            if lang == "pl":
                return (
                    f"Cześć! Twoja **{meta.name}** osiągnęła R²={r2 * 100:.1f}%. "
                    f"Co chciałbyś zrozumieć — metryki, równanie czy wpływ parametrów?"
                )
            return (
                f"Hi! Your **{meta.name}** achieved R²={r2 * 100:.1f}%. "
                f"What would you like to understand — metrics, the equation, or parameter effects?"
            )

        elif meta.task == "classification":
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(run_y, y_pred)
            k = last_params.get("n_neighbors")
            if k is not None:  # KNN
                if lang == "pl":
                    return (
                        f"Cześć! Twój **{meta.name}** (K={k}) osiągnął Accuracy={acc * 100:.1f}%. "
                        f"Możemy porozmawiać o wpływie K na granicę decyzyjną, metrykach "
                        f"odległości lub o tym, co oznaczają precision i recall."
                    )
                return (
                    f"Hi! Your **{meta.name}** (K={k}) achieved Accuracy={acc * 100:.1f}%. "
                    f"We can talk about K's effect on the decision boundary, distance metrics, "
                    f"or what precision and recall mean."
                )
            else:  # Logistic Regression
                if lang == "pl":
                    return (
                        f"Cześć! Twoja **{meta.name}** osiągnęła Accuracy={acc * 100:.1f}%. "
                        f"Chętnie wyjaśnię macierz pomyłek, różnicę między precision a recall "
                        f"lub wpływ parametru C na regularyzację."
                    )
                return (
                    f"Hi! Your **{meta.name}** achieved Accuracy={acc * 100:.1f}%. "
                    f"I can explain the confusion matrix, the difference between precision and recall, "
                    f"or how parameter C controls regularization."
                )
    except Exception:
        pass

    if lang == "pl":
        return f"Cześć! Pracujesz z **{meta.name}**. O co chciałbyś zapytać?"
    return f"Hi! You're working with **{meta.name}**. What would you like to know?"


# ---------------------------------------------------------------------------
# Topic validation
# ---------------------------------------------------------------------------

def validate_question_local(question: str) -> bool:
    """Layer 1: disabled — Layer 3 system prompt acts as the off-topic guard."""
    return True


def validate_question_llm(question: str, client: genai.Client) -> bool:
    """
    Layer 2: single Gemini call to classify the question.
    Called only when validate_question_local() returns False.
    Returns True (allow) on any error — Layer 3 system prompt acts as final guard.
    """
    prompt = (
        "Is the following question related to machine learning, statistics, "
        "data analysis, or data science?\n"
        'Respond ONLY with valid JSON: {"ml_related": true} or {"ml_related": false}\n'
        f'Question: "{question}"'
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        return bool(data.get("ml_related", False))
    except Exception:
        return True  # Fallback: allow, system prompt is the final guard


# ---------------------------------------------------------------------------
# Main response
# ---------------------------------------------------------------------------

def get_assistant_response(
    question: str,
    history: list,
    model_context: str,
    lang: str,
    client: genai.Client,
) -> str:
    """
    Main Gemini call with rolling conversation history and model context.
    Returns assistant response as a plain string.
    """
    template = SYSTEM_PROMPT_TEMPLATE_PL if lang == "pl" else SYSTEM_PROMPT_TEMPLATE_EN
    system_instruction = template.format(model_context=model_context)

    # Build history in google-genai Content format
    gemini_history = []
    for msg in history[-18:]:  # rolling window: max 18 past + current question = 19
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )

    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            max_output_tokens=1024,
            top_p=0.8,
        ),
        history=gemini_history,
    )
    response = chat.send_message(question)
    return response.text
