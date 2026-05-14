# KCK-cyber-trener

Trener Personalny - Analiza Przysiadów Sumo

Projekt inteligentnego asystenta treningowego wykorzystującego wizję komputerową do monitorowania i korygowania techniki wykonywania przysiadów sumo w czasie rzeczywistym.

## 📋 Opis projektu
System został zaprojektowany jako całkowicie lokalna aplikacja (Edge Computing), działająca na procesorze (CPU) laptopa bez konieczności wykorzystywania zewnętrznych serwerów czy układów GPU. Program analizuje sylwetkę użytkownika, liczy powtórzenia oraz udziela wskazówek głosowych.

## 🛠️ Stos technologiczny
* **Język:** Python 3.11
* **Silnik wizyjny:** MediaPipe (model BlazePose) – wybrany ze względu na płynność działania na CPU.
* **Grafika i AR:** OpenCV – nakładanie szkieletu, kątomierzy oraz liczników na obraz z kamery.
* **Interfejs użytkownika (GUI):** TK-Inter – lekka, wbudowana biblioteka do obsługi okien.
* **Interfejs głosowy:** pyttsx3 – system Text-to-Speech działający w 100% offline.
* **Statystyki:** Matplotlib – generowanie wykresów postępu na podstawie danych historycznych.
* **Baza danych:** Pliki lokalne CSV/JSON – prosta forma przechowywania historii treningów bez serwera SQL.

## 🚀 Funkcje systemu
1.  **Analiza postawy (Sumo Squat):** Weryfikacja szerokości rozstawienia stóp oraz głębokości miednicy na podstawie postprocessingu współrzędnych stawów.
2.  **Podstawowe AR:** Wizualizacja szkieletu użytkownika oraz graficznych kątomierzy na kolanach i biodrach.
3.  **Inteligentny Trener:** Automatyczne planowanie celów (progresja powtórzeń) w oparciu o logikę `if-else` analizującą historię z CSV.
4.  **Wsparcie głosowe:** Komendy i liczenie powtórzeń w czasie rzeczywistym.
5.  **Personalizacja:** Zapisywanie indywidualnej głębokości przysiadu podczas sesji kalibracyjnej.

## ⚙️ Wymagania i Instalacja
Zalecane środowisko: **Python 3.11.x**.

1. Sklonuj repozytorium lub pobierz pliki projektu.
2. Stwórz i aktywuj środowisko wirtualne:
   ```bash
   py -3.11 -m venv .venv
   .\\.venv\\Scripts\\activate
3. Zainstaluj wymagane pakiety z pliku zależności:

   ```bash
   pip install -r requirements.txt
4. Aby uruchomić aplikację, upewnij się, że Twoje środowisko .venv jest aktywne, a następnie wpisz w terminalu:

   ```bash
   python main.py
