import cv2
import time
import threading
from gaze_tracking import GazeTracking

class EyeTracker:
    def _init_(self):
        """Initialise le suivi oculaire avec la webcam."""
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        self.running = False
        self.data_points = []
        self.thread = None
        # Données calculées
        self.fixations = []
        self.saccades = []
        self.results = {}  # pour stocker les métriques finales

    def start(self):
        """Démarre le suivi du regard dans un thread séparé."""
        if not self.webcam.isOpened():
            raise Exception("Webcam non accessible")
        self.running = True
        self.start_time = time.time()
        # Lancer le thread de capture
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Boucle interne de capture des frames webcam."""
        while self.running:
            ret, frame = self.webcam.read()
            if not ret:
                break
            self.gaze.refresh(frame)
            # Récupérer données brutes
            ts = time.time() - self.start_time
            lp = self.gaze.pupil_left_coords()
            rp = self.gaze.pupil_right_coords()
            hr = self.gaze.horizontal_ratio()
            vr = self.gaze.vertical_ratio()
            self.data_points.append((ts, lp, rp, hr, vr))
            # Optionnel: on pourrait traiter en temps réel ici,
            # mais il est souvent préférable de stocker puis post-traiter
            time.sleep(0.01)  # petite pause pour éviter 100% CPU, si nécessaire

    def stop(self):
        """Arrête le suivi et analyse les données enregistrées."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.webcam.release()
        # Analyser les données brutes pour extraire fixations/saccades
        self._analyze_data()

    def _analyze_data(self):
        """Analyse les data_points pour calculer fixations, saccades, etc."""
        # Analyse des fixations et saccades (simplifiée pour l’exemple)
        # Initialiser dernière position et début de fixation
        if not self.data_points:
            return
        FIX_THRESHOLD = 0.02  # seuil sur ratio pour considérer une fixation (exemple)
        last_fix_start = self.data_points[0][0]
        last_fix_pos = (self.data_points[0][3], self.data_points[0][4])  # (hr, vr) au début
        fix_point_count = 1
        for i in range(1, len(self.data_points)):
            t, lp, rp, hr, vr = self.data_points[i]
            # distance en terme de ratio du regard (norme L2)
            dist = ((hr - last_fix_pos[0])*2 + (vr - last_fix_pos[1])*2)*0.5
            if dist < FIX_THRESHOLD:
                # toujours dans la même fixation
                fix_point_count += 1
            else:
                # fin de fixation précédente
                fix_end_time = self.data_points[i-1][0]
                duration = fix_end_time - last_fix_start
                fix_avg_pos = last_fix_pos  # approx : centre de fixation = dernier pos connu
                self.fixations.append((last_fix_start, fix_end_time, duration, fix_avg_pos))
                # début d’une nouvelle fixation
                last_fix_start = t
                last_fix_pos = (hr, vr)
                fix_point_count = 1
                # enregistrer la saccade qui a eu lieu
                prev_pos = fix_avg_pos
                new_pos = (hr, vr)
                # déterminer direction de saccade (gauche/droite)
                direction = "right" if new_pos[0] < prev_pos[0] else "left"
                # (Note: si horizontal_ratio plus faible => regard plus à droite,
                # car 0.0 = extrême droite. On inverse la logique ici pour l’utilisateur)
                self.saccades.append((prev_pos, new_pos, direction))
        # On pourrait ajouter la dernière fixation en cours si besoin

        # Calculer les métriques globales
        total_time = self.data_points[-1][0]
        total_words = len(self.text.split()) if hasattr(self, 'text') else None
        wpm = None
        if total_words:
            wpm = (total_words / (total_time/60))  # mots/min
        regressions = sum(1 for s in self.saccades if s[2] == "left")
        avg_fix_dur = (sum(f[2] for f in self.fixations) / len(self.fixations)) if self.fixations else 0
        # Stocker résultats
        self.results = {
            "total_time": total_time,
            "wpm": wpm,
            "fixation_count": len(self.fixations),
            "avg_fixation_duration": avg_fix_dur,
            "saccade_count": len(self.saccades),
            "regression_count": regressions
            # on pourra ajouter plus (distribution etc.)
        }

    def set_text(self, text):
        """Optionnel: fournir le texte en cours (pour le décompte de mots notamment)."""
        self.text = text

    def get_report(self):
        """Génère un rapport textuel des résultats."""
        if not self.results:
            return "No data."
        r = self.results
        report_lines = []
        report_lines.append(f"Reading Time: {r['total_time']:.1f} sec")
        if r.get('wpm'):
            report_lines.append(f"Reading Speed: {r['wpm']:.1f} words/min")
        report_lines.append(f"Total Fixations: {r['fixation_count']} (avg duration {r['avg_fixation_duration']*1000:.0f} ms)")
        report_lines.append(f"Total Saccades: {r['saccade_count']} (of which regressions: {r['regression_count']})")
        # ... autres détails
        return "\n".join(report_lines)
    

