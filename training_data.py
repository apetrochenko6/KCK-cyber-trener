import csv
import os
from datetime import datetime


class TrainingHistoryGuided:
    def __init__(self, filename="sumo_guided_history.csv"):
        self.history_file = filename
        self._initialize_csv()

    def _initialize_csv(self):
        if not os.path.exists(self.history_file):
            header = ['Date', 'Session Name', 'Set ID', 'Target Reps', 'Logged Reps', 'Set Completed']
            try:
                with open(self.history_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
            except IOError as e:
                print(f"Błąd inicjalizacji pliku historii: {e}")

    def get_dashboard_summary(self):
        total_reps = 0
        total_sets = 0
        sessions_map = {}

        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        date = row['Date'].split(' ')[0]
                        name = row['Session Name']
                        reps = int(row['Logged Reps'])

                        total_reps += reps
                        total_sets += 1

                        key = (date, name)
                        if key not in sessions_map:
                            sessions_map[key] = {'date': date, 'name': name, 'reps': 0, 'sets': 0}

                        sessions_map[key]['reps'] += reps
                        sessions_map[key]['sets'] += 1
            except Exception as e:
                print(f"Błąd odczytu: {e}")

        sessions_list = sorted(sessions_map.values(), key=lambda x: x['date'], reverse=True)
        avg_reps = int(total_reps / total_sets) if total_sets > 0 else 0
        last_session = sessions_list[0]['name'] if sessions_list else "Brak"

        return {
            "total_reps": total_reps,
            "avg_reps_per_set": avg_reps,
            "last_session": last_session,
            "sessions_list": sessions_list
        }

    def save_complete_session(self, session_data):
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.history_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for set_data in session_data['sets_list']:
                    row = [
                        date_str,
                        session_data['session_name'],
                        set_data['set_id'],
                        set_data['target_reps'],
                        set_data['logged_reps'],
                        set_data['completed']
                    ]
                    writer.writerow(row)
            return True
        except IOError:
            return False

    def get_structured_session_string(self, reps, sets):
        return f"Trening zakończony. Wykonałeś {sets} serii i łącznie {reps} powtórzeń."