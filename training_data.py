import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tkinter import messagebox


class TrainingData:
    def __init__(self, history_file="treningi_historia.csv"):
        self.history_file = history_file
        self.header = ["Data", "Powtorzenia", "Serie"]

    def _read_rows(self):
        if not os.path.isfile(self.history_file):
            return []
        rows = []
        with open(self.history_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    reps = int(row.get("Powtorzenia", 0))
                    sets = int(row.get("Serie") or 1)
                    date = row.get("Data", "Brak")
                    rows.append({"date": date, "reps": reps, "sets": max(sets, 1)})
                except (ValueError, TypeError):
                    pass
        return rows

    def _rewrite_rows(self, rows):
        with open(self.history_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            for row in rows:
                writer.writerow([row["date"], row["reps"], row["sets"]])

    def save_to_csv(self, count, sets_count=1):
        rows = self._read_rows()
        rows.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "reps": int(count),
            "sets": max(int(sets_count), 1)
        })
        self._rewrite_rows(rows)

    def get_progression_message(self, current_counter):
        rows = self._read_rows()
        if len(rows) >= 3 and all(row["reps"] / row["sets"] >= 10 for row in rows[-3:]):
            return "Świetna robota. Osiągnąłeś stabilne 10 powtórzeń na serię. Twój nowy cel to 12."
        return f"Trening zapisany. Wykonałeś {current_counter} powtórzeń."

    def show_stats(self):
        rows = self._read_rows()
        if not rows:
            messagebox.showwarning("Brak", "Brak poprawnych danych do wykresu.")
            return
        counts = [row["reps"] for row in rows]
        plt.figure(figsize=(8, 4))
        plt.plot(counts, marker="o")
        plt.title("Postępy treningowe")
        plt.xlabel("Numer treningu")
        plt.ylabel("Liczba powtórzeń")
        plt.grid(True)
        plt.show()

    def get_dashboard_summary(self):
        rows = self._read_rows()
        if not rows:
            return {
                "total_reps": 0,
                "avg_reps_per_set": 0,
                "last_session": "Brak",
                "sessions_list": []
            }
        total_reps = sum(row["reps"] for row in rows)
        total_sets = sum(row["sets"] for row in rows)
        sessions = []
        for index, row in enumerate(reversed(rows), start=1):
            sessions.append({
                "date": row["date"],
                "name": f"Sesja {len(rows) - index + 1}",
                "reps": row["reps"]
            })
        return {
            "total_reps": total_reps,
            "avg_reps_per_set": round(total_reps / total_sets, 1),
            "last_session": rows[-1]["date"],
            "sessions_list": sessions
        }
