Tkinter : eye_tracker = EyeTracker()

# Au moment de lancer le test:
eye_tracker.set_text(text_normal)  # texte normal par ex.
eye_tracker.start()
# ... afficher le texte normal, laisser l'utilisateur lire ...
# A la fin de la lecture du texte normal:
eye_tracker.stop()
report_normal = eye_tracker.get_report()

# Puis même chose pour texte morphé, éventuellement recréer un EyeTracker ou réinitialiser
eye_tracker2 = EyeTracker()
eye_tracker2.set_text(text_morphe)
eye_tracker2.start()
# ... afficher texte morphé ...
eye_tracker2.stop()
report_morphe = eye_tracker2.get_report()

# Comparer les deux rapports ou données:
comparison = compare_results(eye_tracker.results, eye_tracker2.results)
show_report_on_ui(report_normal, report_morphe, comparison)