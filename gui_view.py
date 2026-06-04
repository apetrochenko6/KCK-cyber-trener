import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk


class TrainerGuiView:
    def __init__(self, window, start_cmd, stop_cmd, log_cmd, save_cmd, discard_cmd, tutorial_cmd):
        self.window = window
        self.cmds = {
            'start': start_cmd,
            'stop': stop_cmd,
            'save': save_cmd,
            'log': log_cmd,
            'discard': discard_cmd,
            'tutorial': tutorial_cmd
        }

        # Konfiguracja silnika CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.dark_bg = "#242424"
        self.light_bg = "#ebebeb"
        self._set_root_background(self.dark_bg)

        self.fonts = {
            "title": ("Segoe UI", 28, "bold"),
            "header": ("Segoe UI", 16, "bold"),
            "stat_val": ("Segoe UI", 36, "bold"),
            "stat_label": ("Segoe UI", 12),
            "btn": ("Segoe UI", 14, "bold")
        }
        self._create_layout()

    def _create_layout(self):
        # Konfiguracja siatki (Grid) - Blokujemy lewą kolumnę na sztywne 340px
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, minsize=340, weight=0)
        self.window.grid_columnconfigure(1, weight=1)

        # --- PANEL BOCZNY (Sidebar) ---
        self.sidebar = ctk.CTkFrame(self.window, corner_radius=0, fg_color=("#f4f4f4", "#242424"))
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="CYBERTRAINER", font=self.fonts["title"], text_color="#00E676").pack(pady=(40, 5))
        ctk.CTkLabel(self.sidebar, text="SUMO SQUAT ANALYZER", font=("Segoe UI", 10), text_color="gray").pack()

        # BLOK OPISU APLIKACJI (Wypełnienie wolnej przestrzeni)
        opis_tekst = (
            "Inteligentny asystent treningowy\n"
            "działający całkowicie w trybie offline.\n\n"
            "Analizuje postawę, liczy powtórzenia\n"
            "i pomaga poprawiać technikę\n"
            "przysiadu sumo w czasie rzeczywistym."
        )
        self.lbl_description = ctk.CTkLabel(self.sidebar, text=opis_tekst, font=self.fonts["stat_label"],
                                            text_color="gray", justify="center")
        self.lbl_description.pack(pady=(30, 20))

        # Separator wizualny
        self.separator = ctk.CTkFrame(self.sidebar, height=2, fg_color=("gray80", "gray20"))
        self.separator.pack(fill="x", padx=40, pady=(0, 10))

        # --- KONTROLKI KREATORA SESJI (WŁĄCZANE PODCZAS RUCHU) ---
        self.wizard_controls = ctk.CTkFrame(self.sidebar, fg_color="transparent")

        ctk.CTkLabel(self.wizard_controls, text="USTAWIENIA SERII", font=self.fonts["header"]).pack(pady=(20, 10))
        ctk.CTkLabel(self.wizard_controls, text="CEL POWTÓRZEŃ:", font=self.fonts["stat_label"],
                     text_color="gray").pack(pady=(10, 0))

        self.target_reps_var = ctk.IntVar(value=10)
        self.slider = ctk.CTkSlider(self.wizard_controls, from_=1, to=20, variable=self.target_reps_var,
                                    number_of_steps=19, button_color="#00E676", progress_color="#00E676")
        self.slider.pack(pady=10)
        self.slider_val_label = ctk.CTkLabel(self.wizard_controls, textvariable=self.target_reps_var,
                                             font=self.fonts["btn"])
        self.slider_val_label.pack()

        ctk.CTkButton(self.wizard_controls, text="ZALOGUJ SERIĘ", command=self.cmds['log'], font=self.fonts["btn"],
                      fg_color="#2196F3", height=40).pack(pady=(30, 10), padx=40, fill="x")
        ctk.CTkButton(self.wizard_controls, text="ZAKOŃCZ I ZAPISZ", command=self.cmds['stop'], font=self.fonts["btn"],
                      fg_color="#FF5252", hover_color="#D32F2F", height=40).pack(pady=10, padx=40, fill="x")

        # --- DOLNA SEKCJA PANELU ---
        self.bottom_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.bottom_frame.pack(side="bottom", fill="x", pady=20)

        self.theme_switch = ctk.CTkSwitch(self.bottom_frame, text="Tryb Ciemny", command=self.toggle_theme,
                                          progress_color="#00E676")
        self.theme_switch.pack(pady=(0, 20))
        self.theme_switch.select()

        ctk.CTkLabel(self.bottom_frame, text="Projekt Zespołowy - KCK", font=("Segoe UI", 10), text_color="gray").pack()

        # --- GŁÓWNY OBSZAR WYŚWIETLANIA ---
        self.main_content = ctk.CTkFrame(self.window, corner_radius=20, fg_color=("#f4f4f4", "#242424"))
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.screen_dashboard = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.screen_session = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.screen_summary = ctk.CTkFrame(self.main_content, fg_color="transparent")

        self._build_dashboard()
        self._build_session()
        self._build_summary()

    def _build_dashboard(self):
        ctk.CTkButton(self.screen_dashboard, text="NOWA SESJA TRENINGOWA", font=self.fonts["title"],
                      command=self.cmds['start'], height=60, fg_color="#00E676", hover_color="#00C853",
                      text_color="black").pack(pady=(40, 10))

        # PRZYCISK TUTORIAL (Wywołuje pustą funkcję)
        ctk.CTkButton(self.screen_dashboard, text="TUTORIAL", font=self.fonts["btn"], command=self.cmds['tutorial'],
                      height=40, fg_color="#2196F3", hover_color="#1976D2", text_color="white").pack(pady=(0, 40))

        stats_frame = ctk.CTkFrame(self.screen_dashboard, fg_color="transparent")
        stats_frame.pack(fill="x", padx=20)

        self.stat_total = self._create_stat_card(stats_frame, "CAŁKOWITE POWT.", "0", "#00E676")
        self.stat_total.pack(side="left", expand=True, padx=10)

        self.stat_avg = self._create_stat_card(stats_frame, "ŚREDNIA NA SERIĘ", "0", "#2196F3")
        self.stat_avg.pack(side="left", expand=True, padx=10)

        self.stat_last = self._create_stat_card(stats_frame, "OSTATNIA SESJA", "Brak", "gray")
        self.stat_last.pack(side="left", expand=True, padx=10)

        ctk.CTkLabel(self.screen_dashboard, text="HISTORIA SESJI:", font=self.fonts["header"], text_color="gray").pack(
            anchor="w", padx=30, pady=(40, 10))
        self.history_scroll = ctk.CTkScrollableFrame(self.screen_dashboard, fg_color="transparent")
        self.history_scroll.pack(fill="both", expand=True, padx=20, pady=10)

    def _create_stat_card(self, parent, title, val, color):
        frame = ctk.CTkFrame(parent, corner_radius=15)
        ctk.CTkLabel(frame, text=title, font=self.fonts["stat_label"], text_color="gray").pack(pady=(15, 0), padx=20,
                                                                                               anchor="w")
        lbl = ctk.CTkLabel(frame, text=val, font=self.fonts["stat_val"], text_color=color)
        lbl.pack(pady=(0, 15), padx=20, anchor="w")
        frame.val_label = lbl
        return frame

    def _build_session(self):
        live_stats_frame = ctk.CTkFrame(self.screen_session, fg_color="transparent")
        live_stats_frame.pack(fill="x", padx=20, pady=(10, 0))

        # Etykieta licznika powtórzeń (Zielona)
        ctk.CTkLabel(live_stats_frame, text="POWTÓRZENIA:", font=self.fonts["stat_label"], text_color="gray").pack(side="left", padx=(0, 10))
        self.counter_label = ctk.CTkLabel(live_stats_frame, text="0", font=self.fonts["stat_val"], text_color="#00E676")
        self.counter_label.pack(side="left")

        # Etykieta kąta docelowego (Niebieska)
        self.target_label = ctk.CTkLabel(live_stats_frame, text="CALIB", font=self.fonts["stat_val"], text_color="#2196F3")
        self.target_label.pack(side="right")
        ctk.CTkLabel(live_stats_frame, text="CEL:", font=self.fonts["stat_label"], text_color="gray").pack(side="right", padx=(10, 10))

        self.video_label = tk.Label(self.screen_session, bg="black")
        self.video_label.pack(expand=True, fill="both", padx=20, pady=(10, 10))

        self.status_label = ctk.CTkLabel(self.screen_session, text="Gotowy.", font=self.fonts["header"],
                                         fg_color=("gray80", "gray20"), corner_radius=10, height=40)
        self.status_label.pack(fill="x", padx=20, pady=5)

        self.live_sets_scroll = ctk.CTkScrollableFrame(self.screen_session, height=120, fg_color="transparent")
        self.live_sets_scroll.pack(fill="x", padx=20, pady=(5, 20))

    def _build_summary(self):
        ctk.CTkLabel(self.screen_summary, text="PODSUMOWANIE SESJI", font=self.fonts["title"]).pack(pady=40)

        stats_frame = ctk.CTkFrame(self.screen_summary, fg_color="transparent")
        stats_frame.pack(fill="x", padx=40)

        self.sum_total = self._create_stat_card(stats_frame, "ZROBIONE POWTÓRZENIA", "0", "#00E676")
        self.sum_total.pack(side="left", expand=True, padx=10)

        self.sum_sets = self._create_stat_card(stats_frame, "ZROBIONE SERIE", "0", "#2196F3")
        self.sum_sets.pack(side="left", expand=True, padx=10)

        self.summary_scroll = ctk.CTkScrollableFrame(self.screen_summary, fg_color="transparent")
        self.summary_scroll.pack(fill="both", expand=True, padx=40, pady=20)

        btn_frame = ctk.CTkFrame(self.screen_summary, fg_color="transparent")
        btn_frame.pack(fill="x", padx=40, pady=20)

        ctk.CTkButton(btn_frame, text="ZAPISZ DO HISTORII", command=self.cmds['save'], font=self.fonts["btn"],
                      fg_color="#00E676", hover_color="#00C853", text_color="black", height=50).pack(side="left",
                                                                                                     expand=True,
                                                                                                     padx=10)
        ctk.CTkButton(btn_frame, text="ODRZUĆ", command=self.cmds['discard'], font=self.fonts["btn"],
                      fg_color="#FF5252", hover_color="#D32F2F", height=50).pack(side="left", expand=True, padx=10)

    def toggle_theme(self):
        if self.theme_switch.get() == 1:
            ctk.set_appearance_mode("dark")
            self._set_root_background(self.dark_bg)
            self.theme_switch.configure(text="Tryb Ciemny")
        else:
            ctk.set_appearance_mode("light")
            self._set_root_background(self.light_bg)
            self.theme_switch.configure(text="Tryb Jasny")


    def show_dashboard(self, summary_data=None):
        self.screen_session.pack_forget()
        self.screen_summary.pack_forget()
        self.wizard_controls.pack_forget()
        self.screen_dashboard.pack(fill="both", expand=True)

        if summary_data:
            self.stat_total.val_label.configure(text=str(summary_data['total_reps']))
            self.stat_avg.val_label.configure(text=str(summary_data['avg_reps_per_set']))
            self.stat_last.val_label.configure(text=str(summary_data['last_session']))

            for widget in self.history_scroll.winfo_children():
                widget.destroy()

            for session in summary_data['sessions_list'][:10]:
                frame = ctk.CTkFrame(self.history_scroll, corner_radius=10)
                frame.pack(fill="x", pady=5)
                ctk.CTkLabel(frame, text=f"{session['date']} | {session['name']}", font=self.fonts["btn"]).pack(
                    side="left", padx=20, pady=15)
                ctk.CTkLabel(frame, text=f"{session['reps']} powtórzeń", text_color="gray").pack(side="right", padx=20)

    def show_session_running(self):
        self.screen_dashboard.pack_forget()
        self.screen_summary.pack_forget()
        self.wizard_controls.pack(fill="both", expand=True, pady=10)
        self.screen_session.pack(fill="both", expand=True)

        for widget in self.live_sets_scroll.winfo_children():
            widget.destroy()

    def show_summary(self, session_data):
        self.screen_session.pack_forget()
        self.wizard_controls.pack_forget()
        self.screen_summary.pack(fill="both", expand=True)

        self.sum_total.val_label.configure(text=str(session_data['total_reps']))
        self.sum_sets.val_label.configure(text=str(len(session_data['sets_list'])))

        for widget in self.summary_scroll.winfo_children():
            widget.destroy()

        for s in session_data['sets_list']:
            frame = ctk.CTkFrame(self.summary_scroll, corner_radius=10)
            frame.pack(fill="x", pady=5)
            ctk.CTkLabel(frame, text=f"Seria {s['set_id']} (Cel: {s['target_reps']})", font=self.fonts["btn"]).pack(
                side="left", padx=20, pady=15)
            ctk.CTkLabel(frame, text=f"{s['logged_reps']} powtórzeń", text_color="gray").pack(side="right", padx=20)

    def update_video(self, img):
        imgtk = ImageTk.PhotoImage(img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def update_status(self, text):
        self.status_label.configure(text=text)

    def update_live_sets(self, sets_list):
        for widget in self.live_sets_scroll.winfo_children():
            widget.destroy()
        for s in sets_list:
            frame = ctk.CTkFrame(self.live_sets_scroll, corner_radius=10)
            frame.pack(fill="x", pady=5)
            ctk.CTkLabel(frame, text=f"Seria {s['set_id']} (Cel: {s['target_reps']})", font=self.fonts["btn"]).pack(
                side="left", padx=20, pady=10)
            ctk.CTkLabel(frame, text=f"{s['logged_reps']} powtórzeń", text_color="gray").pack(side="right", padx=20)

    def get_target_reps(self):
        return self.target_reps_var.get()

    def update_counter(self, value):
        if hasattr(self, 'counter_label'):
            self.counter_label.configure(text=str(value))

    def update_target(self, text):
        if hasattr(self, 'target_label'):
            self.target_label.configure(text=str(text))

    def _set_root_background(self, color):
        try:
            self.window.configure(fg_color=color)
        except tk.TclError:
            self.window.configure(bg=color)