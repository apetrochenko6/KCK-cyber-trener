import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import messagebox

class TrainingData:
    """Klasa zarządzająca historią treningów (CSV) oraz wizualizacją."""
    def __init__(self, history_file="treningi_historia.csv"):
        self.history_file = history_file

    def save_to_csv(self, count):
        file_exists = os.path.isfile(self.history_file)
        with open(self.history_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Data", "Powtorzenia"])
            writer.writerow([str(np.datetime64("now")), count])

    def get_progression_message(self, current_counter):
        if not os.path.isfile(self.history_file):
            return f"Trening zapisany. Wykonałeś {current_counter} powtórzeń."

        counts = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    counts.append(int(row["Powtorzenia"]))
                except ValueError:
                    pass

        if len(counts) >= 3 and all(c >= 10 for c in counts[-3:]):
            return "Świetna robota. Osiągnąłeś stabilne 10 powtórzeń. Twój nowy cel to 12."
        return f"Trening zapisany. Wykonałeś {current_counter} powtórzeń."

    def show_stats(self):
        if not os.path.isfile(self.history_file):
            messagebox.showwarning("Brak", "Brak danych.")
            return

        counts = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    counts.append(int(row["Powtorzenia"]))
                except ValueError:
                    pass

        if not counts:
            messagebox.showwarning("Brak", "Brak poprawnych danych do wykresu.")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(counts, marker="o")
        plt.title("Postępy treningowe")
        plt.xlabel("Numer treningu")
        plt.ylabel("Liczba powtórzeń")
        plt.grid(True)
        plt.show()