# KCK-cyber-trener

import os

readme_content = """# Trener Personalny - Analiza Przysiadów Sumo

Projekt inteligentnego asystenta treningowego wykorzystującego wizję komputerową do monitorowania i korygowania techniki wykonywania przysiadów sumo w czasie rzeczywistym[cite: 1, 2].

## 📋 Opis projektu
System został zaprojektowany jako całkowicie lokalna aplikacja (Edge Computing), działająca na procesorze (CPU) laptopa bez konieczności wykorzystywania zewnętrznych serwerów czy układów GPU[cite: 5, 6]. Program analizuje sylwetkę użytkownika, liczy powtórzenia oraz udziela wskazówek głosowych[cite: 14, 21].

## 🛠️ Stos technologiczny
* **Język:** Python 3.11 / 3.12
* **Silnik wizyjny:** MediaPipe (model BlazePose) – wybrany ze względu na płynność działania na CPU[cite: 2, 21].
* **Grafika i AR:** OpenCV – nakładanie szkieletu, kątomierzy oraz liczników na obraz z kamery[cite: 16, 17].
* **Interfejs użytkownika (GUI):** TK-Inter – lekka, wbudowana biblioteka do obsługi okien[cite: 12, 13].
* **Interfejs głosowy:** pyttsx3 – system Text-to-Speech działający w 100% offline[cite: 14].
* **Statystyki:** Matplotlib – generowanie wykresów postępu na podstawie danych historycznych[cite: 10, 11].
* **Baza danych:** Pliki lokalne CSV/JSON – prosta forma przechowywania historii treningów bez serwera SQL[cite: 9].

## 🚀 Funkcje systemu
1.  **Analiza postawy (Sumo Squat):** Weryfikacja szerokości rozstawienia stóp oraz głębokości miednicy na podstawie postprocessingu współrzędnych stawów[cite: 21].
2.  **Podstawowe AR:** Wizualizacja szkieletu użytkownika oraz graficznych kątomierzy na kolanach i biodrach[cite: 17].
3.  **Inteligentny Trener:** Automatyczne planowanie celów (progresja powtórzeń) w oparciu o logikę `if-else` analizującą historię z CSV[cite: 24, 25].
4.  **Wsparcie głosowe:** Komendy i liczenie powtórzeń w czasie rzeczywistym[cite: 14].
5.  **Personalizacja:** Zapisywanie indywidualnej głębokości przysiadu podczas sesji kalibracyjnej[cite: 22].

## ⚙️ Wymagania i Instalacja
Zalecane środowisko: **Python 3.11.x**.

1. Sklonuj repozytorium lub pobierz pliki projektu.
2. Stwórz i aktywuj środowisko wirtualne:
   ```bash
   python -m venv .venv
   .\\.venv\\Scripts\\activate