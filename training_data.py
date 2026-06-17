import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tkinter import messagebox


class TrainingData:
    def __init__(self, history_file="treningi_historia.csv"):
        if os.path.isabs(history_file):
            self.history_file = history_file
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.history_file = os.path.join(base_dir, history_file)

    def save_to_csv(self, count):
        count = int(count)
        file_exists = os.path.isfile(self.history_file)

        with open(self.history_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(self.history_file) == 0:
                writer.writerow(["Data", "Powtorzenia"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), count])

    def _read_history_rows(self):
        if not os.path.isfile(self.history_file):
            return []

        with open(self.history_file, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(2048)
            f.seek(0)

            delimiter = ";"
            if "," in sample and sample.count(",") >= sample.count(";"):
                delimiter = ","

            reader = csv.DictReader(f, delimiter=delimiter)
            rows = []

            for row in reader:
                if not row:
                    continue

                date = (
                    row.get("Data")
                    or row.get("data")
                    or row.get("Date")
                    or row.get("date")
                    or "Brak"
                )

                reps_raw = (
                    row.get("Powtorzenia")
                    or row.get("Powtórzenia")
                    or row.get("powtorzenia")
                    or row.get("powtórzenia")
                    or row.get("Reps")
                    or row.get("reps")
                    or row.get("Liczba powtórzeń")
                    or row.get("Liczba powtorzen")
                    or "0"
                )

                try:
                    reps = int(float(str(reps_raw).strip().replace(",", ".")))
                except (TypeError, ValueError):
                    continue

                rows.append({"date": str(date), "reps": reps})

            return rows

    def get_progression_message(self, current_counter):
        rows = self._read_history_rows()
        counts = [row["reps"] for row in rows]

        if len(counts) >= 3 and all(c >= 10 for c in counts[-3:]):
            return "Świetna robota. Osiągnąłeś stabilne 10 powtórzeń. Twój nowy cel to 12."

        return f"Trening zapisany. Wykonałeś {current_counter} powtórzeń."

    def show_stats(self):
        rows = self._read_history_rows()

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
        rows = self._read_history_rows()

        if not rows:
            return {
                "total_reps": 0,
                "avg_reps_per_set": 0,
                "last_session": "Brak",
                "sessions_list": []
            }

        total_reps = sum(row["reps"] for row in rows)
        avg_reps = round(total_reps / len(rows), 1)
        last_session = rows[-1]["date"]

        sessions_list = [
            {
                "date": row["date"],
                "name": f"Trening {index}",
                "reps": row["reps"]
            }
            for index, row in enumerate(rows, start=1)
        ]

        return {
            "total_reps": total_reps,
            "avg_reps_per_set": avg_reps,
            "last_session": last_session,
            "sessions_list": list(reversed(sessions_list))
        }
