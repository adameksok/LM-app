"""
test_gemini.py — szybki test połączenia z Gemini API
Uruchom: python test_gemini.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "")

print("=" * 50)
print("TEST POŁĄCZENIA Z GEMINI API")
print("=" * 50)

# 1. Sprawdź klucz
if not api_key:
    print("BLAD: Brak klucza GEMINI_API_KEY")
    print("  Dodaj do pliku .env: GEMINI_API_KEY=twoj_klucz")
    exit(1)

masked = api_key[:8] + "..." + api_key[-4:]
print(f"  Klucz API: {masked}")

# 2. Importuj SDK
try:
    from google import genai
    print(f"  SDK google-genai: OK (wersja {genai.__version__})")
except ImportError as e:
    print(f"  BLAD importu SDK: {e}")
    print("  Uruchom: pip install google-genai")
    exit(1)

# 3. Inicjalizuj klienta
try:
    client = genai.Client(api_key=api_key)
    print("  Klient Gemini: OK")
except Exception as e:
    print(f"  BLAD inicjalizacji klienta: {e}")
    exit(1)

# 4. Wyslij testowe zapytanie
print()
print("Wysylanie testowego zapytania...")
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Odpowiedz jednym zdaniem po polsku: Co to jest regresja liniowa?",
    )
    print()
    print("ODPOWIEDZ MODELU:")
    print(f"  {response.text.strip()}")
    print()
    print("POLACZENIE DZIALA POPRAWNIE")
except Exception as e:
    err = str(e)
    print(f"  BLAD: {err}")
    if "429" in err or "quota" in err.lower() or "resource" in err.lower():
        print()
        print("  Przekroczono limit API (quota).")
        print("  Darmowy tier: 15 zapytan/minute, 1500/dzien")
        print("  Poczekaj minute i sprobuj ponownie.")
    elif "invalid" in err.lower() or "api_key" in err.lower() or "401" in err:
        print()
        print("  Nieprawidlowy klucz API.")
        print("  Sprawdz klucz na: https://aistudio.google.com/app/apikey")
    elif "not found" in err.lower() or "404" in err:
        print()
        print("  Model nie znaleziony.")
        print("  Sprawdz dostepnosc gemini-2.0-flash w swoim regionie.")
    exit(1)
