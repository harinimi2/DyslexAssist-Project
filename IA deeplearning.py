import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import pygame
import dlib
import math
import random
from scipy.stats import maxwell
import time
import os
import json
from collections import deque
import threading
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Initialisation des paramètres
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
FONT_SIZE = 36
TEXT_MARGIN = 50
GAZE_HISTORY_LENGTH = 15
MODEL_PATH = "dyslexic_morph_model.h5"
USER_PROFILE_PATH = "user_profile.json"
CALIBRATION_DATA_PATH = "calibration_data.json"
EYE_TRACKER = "mediapipe"  # Options: "dlib" ou "mediapipe"

# Classe pour le suivi oculaire
class EyeTracker:
    def __init__(self):
        self.eye_tracker_type = EYE_TRACKER
        
        if self.eye_tracker_type == "dlib":
            # Initialisation du détecteur dlib
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        else:
            # Initialisation de MediaPipe
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            
            # Points d'intérêt pour les yeux
            self.left_eye_landmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
            self.right_eye_landmarks = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        
        self.last_gaze_point = (0, 0)
        self.calibration_matrix = None
        self.calibration_intercept = None
    
    def detect_eyes(self, frame):
        if self.eye_tracker_type == "dlib":
            return self._detect_eyes_dlib(frame)
        else:
            return self._detect_eyes_mediapipe(frame)
    
    def _detect_eyes_dlib(self, frame):
        """Détecte les yeux avec dlib"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None, None
        
        # Prendre le premier visage détecté
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Coordonnées des yeux
        left_eye = []
        right_eye = []
        
        # Points pour l'œil gauche (36-41)
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
        
        # Points pour l'œil droit (42-47)
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))
        
        return left_eye, right_eye
    
    def _detect_eyes_mediapipe(self, frame):
        """Détecte les yeux avec MediaPipe"""
        # Convertir l'image et traiter avec MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Prendre le premier visage détecté
        face_landmarks = results.multi_face_landmarks[0]
        ih, iw, _ = frame.shape
        
        left_eye = []
        right_eye = []
        
        # Extraire les points de l'œil gauche
        for idx in self.left_eye_landmarks:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * iw)
            y = int(landmark.y * ih)
            left_eye.append((x, y))
        
        # Extraire les points de l'œil droit
        for idx in self.right_eye_landmarks:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * iw)
            y = int(landmark.y * ih)
            right_eye.append((x, y))
        
        return left_eye, right_eye
    
    def estimate_gaze(self, eye_points, frame_width, frame_height):
        """Estime la direction du regard avec une méthode améliorée"""
        # Calcul du centre de l'œil
        eye_center = np.mean(eye_points, axis=0)
        
        # Calcul de l'orientation basée sur la forme de l'œil
        if len(eye_points) >= 6:
            # Points caractéristiques pour l'estimation de la direction
            if self.eye_tracker_type == "dlib":
                left_corner = eye_points[0]
                right_corner = eye_points[3]
                top_point = eye_points[1]
                bottom_point = eye_points[5]
            else:
                # Pour MediaPipe, utiliser les points extérieurs
                left_corner = eye_points[0]
                right_corner = eye_points[8]
                top_point = eye_points[12]
                bottom_point = eye_points[4]
            
            # Calcul des vecteurs
            horizontal_vec = np.array(right_corner) - np.array(left_corner)
            vertical_vec = np.array(bottom_point) - np.array(top_point)
            
            # Normalisation
            horizontal_vec = horizontal_vec / np.linalg.norm(horizontal_vec)
            vertical_vec = vertical_vec / np.linalg.norm(vertical_vec)
            
            # Calcul de l'angle de regard
            angle = math.atan2(vertical_vec[1], horizontal_vec[0])
            
            # Calcul de la distance focale (simulée)
            eye_width = np.linalg.norm(np.array(right_corner) - np.array(left_corner))
            distance = 1.0 / (eye_width + 1e-5)  # Éviter la division par zéro
            
            return eye_center, angle, distance
        
        # Méthode de repli si nous n'avons pas assez de points
        dx = eye_center[0] - frame_width / 2
        dy = eye_center[1] - frame_height / 2
        angle = math.atan2(dy, dx)
        distance = math.sqrt(dx**2 + dy**2) / (frame_width / 2)
        
        return eye_center, angle, distance
    
    def calibrate(self, calibration_points, gaze_data):
        """Calcule la matrice de calibration à partir des points de référence"""
        # Préparer les données pour la régression linéaire
        screen_points = np.array(calibration_points)
        gaze_vectors = np.array(gaze_data)
        
        # Calculer la matrice de transformation
        A = np.hstack([gaze_vectors, np.ones((gaze_vectors.shape[0], 1))])
        B = screen_points
        
        # Résoudre le système d'équations avec la méthode des moindres carrés
        X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        
        self.calibration_matrix = X[:-1, :].T
        self.calibration_intercept = X[-1, :]
        
        return residuals
    
    def map_to_screen(self, gaze_vector):
        """Mappe le vecteur de regard aux coordonnées de l'écran"""
        if self.calibration_matrix is None or self.calibration_intercept is None:
            return (0, 0)
        
        # Transformation linéaire: screen_coords = M * gaze_vector + b
        gaze_vector = np.array(gaze_vector)
        screen_coords = np.dot(self.calibration_matrix, gaze_vector) + self.calibration_intercept
        return (screen_coords[0], screen_coords[1])

# Classe principale du système
class DyslexicAdaptationSystem:
    def __init__(self):
        # Initialisation de Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Système d'Adaptation Typographique Dyslexic Morph")
        
        # Initialisation du suivi oculaire
        self.eye_tracker = EyeTracker()
        
        # Initialisation de la webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la webcam")
        
        # Variables d'état
        self.running = True
        self.calibrated = False
        self.wearing_glasses = False
        self.current_text = ""
        self.user_profile = {}
        self.gaze_history = deque(maxlen=GAZE_HISTORY_LENGTH)
        self.fixation_data = []
        self.preference_test_data = []
        self.last_fixation_time = time.time()
        
        # Charger ou générer des données de test
        self.load_or_generate_test_data()
        
        # Police pour l'affichage
        self.font = pygame.font.SysFont('Arial', FONT_SIZE)
        self.small_font = pygame.font.SysFont('Arial', 24)
        self.title_font = pygame.font.SysFont('Arial', 48, bold=True)
        
        # Initialisation du modèle de deep learning
        self.model = self.build_model()
        
        # Calibration initiale
        self.calibrate_system()
    
    def build_model(self):
        """Construit le modèle de deep learning pour la prédiction de transformation"""
        if os.path.exists(MODEL_PATH):
            print("Chargement du modèle pré-entraîné...")
            return tf.keras.models.load_model(MODEL_PATH)
        
        print("Construction d'un nouveau modèle...")
        # Architecture du modèle améliorée
        input_text = layers.Input(shape=(100,), name='text_input')
        input_gaze = layers.Input(shape=(GAZE_HISTORY_LENGTH, 4), name='gaze_input')
        
        # Branche pour le traitement du texte
        x_text = layers.Embedding(input_dim=10000, output_dim=128)(input_text)
        x_text = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x_text)
        x_text = layers.Bidirectional(layers.LSTM(128))(x_text)
        x_text = layers.Dense(128, activation='relu')(x_text)
        x_text = layers.Dropout(0.4)(x_text)
        
        # Branche pour le traitement du suivi oculaire
        x_gaze = layers.Conv1D(64, 3, activation='relu')(input_gaze)
        x_gaze = layers.MaxPooling1D(2)(x_gaze)
        x_gaze = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x_gaze)
        x_gaze = layers.Bidirectional(layers.LSTM(64))(x_gaze)
        x_gaze = layers.Dense(128, activation='relu')(x_gaze)
        x_gaze = layers.Dropout(0.3)(x_gaze)
        
        # Fusion des deux branches
        merged = layers.concatenate([x_text, x_gaze])
        merged = layers.Dense(256, activation='relu')(merged)
        merged = layers.Dropout(0.5)(merged)
        merged = layers.Dense(128, activation='relu')(merged)
        
        # Couche de sortie
        output = layers.Dense(26 * 3, activation='softmax')(merged)  # 3 variantes par lettre
        
        model = models.Model(inputs=[input_text, input_gaze], outputs=output)
        
        # Optimiseur avec taux d'apprentissage adaptatif
        optimizer = optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        return model
    
    def simulate_training_data(self, num_samples=10000):
        """Génère des données d'entraînement simulées avec plus de réalisme"""
        print(f"Génération de {num_samples} échantillons d'entraînement...")
        # Données de texte simulées (plus réalistes)
        text_data = np.zeros((num_samples, 100), dtype=int)
        for i in range(num_samples):
            # Générer une séquence de mots aléatoires
            words = [random.choice(list(self.letter_variants.keys())) for _ in range(20)]
            text = ' '.join(words)
            # Encoder le texte
            encoded = [ord(c) % 10000 for c in text[:100]]
            text_data[i, :len(encoded)] = encoded
        
        # Données de suivi oculaire simulées (positions x,y, durée, angle)
        gaze_data = np.zeros((num_samples, GAZE_HISTORY_LENGTH, 4))
        for i in range(num_samples):
            for j in range(GAZE_HISTORY_LENGTH):
                # Utiliser une distribution plus réaliste
                gaze_data[i, j, 0] = np.random.beta(2, 5)  # position x (gauche-droite)
                gaze_data[i, j, 1] = np.random.beta(3, 3)  # position y (haut-bas)
                
                # Durée de fixation avec distribution de Maxwell
                scale = 0.4 if self.wearing_glasses else 0.3
                gaze_data[i, j, 2] = maxwell.rvs(loc=0.1, scale=scale)  # durée
                
                # Angle avec distribution normale autour du centre
                gaze_data[i, j, 3] = np.random.normal(0, 0.5)  # angle
        
        # Étiquettes simulées (transformations de lettres)
        labels = np.zeros((num_samples, 26 * 3))
        letter_difficulty = {
            'b': 0.85, 'd': 0.82, 'p': 0.80, 'q': 0.78, 
            'a': 0.65, 'e': 0.60, 'o': 0.55, 's': 0.50,
            'm': 0.48, 'n': 0.45, 'u': 0.42, 'v': 0.40,
            'c': 0.35, 'g': 0.33, 'f': 0.30, 'h': 0.28,
            'i': 0.25, 'j': 0.23, 'k': 0.20, 'l': 0.18,
            'r': 0.15, 't': 0.13, 'w': 0.10, 'x': 0.08,
            'y': 0.05, 'z': 0.03
        }
        
        for i in range(num_samples):
            text = ''.join([chr(c % 10000) for c in text_data[i] if c > 0])
            for j, char in enumerate(text[:26]):  # Seulement les premiers caractères
                if char.lower() in letter_difficulty:
                    # Probabilité de transformation basée sur la difficulté
                    prob = letter_difficulty[char.lower()]
                    if np.random.random() < prob:
                        # Choix d'une variante (0 = originale, 1-2 = alternatives)
                        variant = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
                        letter_index = ord(char.lower()) - ord('a')
                        labels[i, letter_index * 3 + variant] = 1
                    else:
                        letter_index = ord(char.lower()) - ord('a')
                        labels[i, letter_index * 3] = 1
        
        return [text_data, gaze_data], labels
    
    def train_model(self, epochs=15):
        """Entraîne le modèle avec des données simulées et callbacks"""
        print("Génération des données d'entraînement...")
        X, y = self.simulate_training_data(10000)
        
        # Séparation des données
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Callbacks pour l'entraînement
        checkpoint = callbacks.ModelCheckpoint(
            MODEL_PATH, 
            save_best_only=True, 
            monitor='val_accuracy', 
            mode='max'
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3, 
            min_lr=1e-6
        )
        
        print("Entraînement du modèle...")
        history = self.model.fit(
            [X_train[0], X_train[1]], 
            y_train,
            epochs=epochs,
            batch_size=64,
            validation_data=([X_val[0], X_val[1]], y_val),
            callbacks=[checkpoint, early_stop, reduce_lr]
        )
        
        print("Sauvegarde du modèle final...")
        self.model.save(MODEL_PATH)
        return history
    
    def calibrate_system(self):
        """Calibre le système pour l'utilisateur"""
        print("Début de la calibration...")
        calibration_points = [
            (SCREEN_WIDTH * 0.2, SCREEN_HEIGHT * 0.2),
            (SCREEN_WIDTH * 0.8, SCREEN_HEIGHT * 0.2),
            (SCREEN_WIDTH * 0.5, SCREEN_HEIGHT * 0.5),
            (SCREEN_WIDTH * 0.2, SCREEN_HEIGHT * 0.8),
            (SCREEN_WIDTH * 0.8, SCREEN_HEIGHT * 0.8)
        ]
        
        calibration_data = []
        gaze_vectors = []
        
        for i, point in enumerate(calibration_points):
            # Afficher le point de calibration
            self.screen.fill((30, 30, 50))
            pygame.draw.circle(self.screen, (0, 200, 255), point, 25)
            
            # Afficher le compteur
            count_text = self.title_font.render(str(5 - i), True, (255, 255, 0))
            self.screen.blit(count_text, (SCREEN_WIDTH//2 - 20, SCREEN_HEIGHT//2 - 50))
            
            instruction = self.font.render("Regardez le point bleu", True, (255, 255, 255))
            self.screen.blit(instruction, (SCREEN_WIDTH//2 - instruction.get_width()//2, SCREEN_HEIGHT - 100))
            
            pygame.display.flip()
            
            # Attendre que l'utilisateur regarde le point
            time.sleep(1)
            
            # Collecter des données de regard
            gaze_samples = []
            start_time = time.time()
            while time.time() - start_time < 3:  # Collecter pendant 3 secondes
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Détecter les yeux
                left_eye, right_eye = self.eye_tracker.detect_eyes(frame)
                if left_eye and right_eye:
                    # Estimer le regard
                    frame_height, frame_width = frame.shape[:2]
                    left_center, left_angle, left_dist = self.eye_tracker.estimate_gaze(left_eye, frame_width, frame_height)
                    right_center, right_angle, right_dist = self.eye_tracker.estimate_gaze(right_eye, frame_width, frame_height)
                    
                    # Moyenne des deux yeux
                    avg_angle = (left_angle + right_angle) / 2
                    avg_dist = (left_dist + right_dist) / 2
                    
                    gaze_samples.append((avg_angle, avg_dist))
            
            if gaze_samples:
                avg_gaze = np.mean(gaze_samples, axis=0)
                gaze_vectors.append(avg_gaze)
                calibration_data.append((point, avg_gaze))
        
        # Calculer la matrice de calibration
        residuals = self.eye_tracker.calibrate(calibration_points, gaze_vectors)
        self.calibrated = True
        print(f"Calibration terminée. Erreur résiduelle: {residuals[0] if len(residuals) > 0 else 'N/A'}")
        
        # Sauvegarder les données de calibration
        with open(CALIBRATION_DATA_PATH, 'w') as f:
            json.dump({
                'calibration_points': calibration_points,
                'gaze_vectors': [list(v) for v in gaze_vectors],
                'residuals': residuals[0] if len(residuals) > 0 else 0.0
            }, f)
        
        # Demander si l'utilisateur porte des lunettes
        self.ask_about_glasses()
    
    def ask_about_glasses(self):
        """Demande à l'utilisateur s'il porte des lunettes"""
        self.screen.fill((30, 30, 50))
        question = self.font.render("Portez-vous des lunettes? (O/N)", True, (255, 255, 255))
        self.screen.blit(question, (SCREEN_WIDTH//2 - question.get_width()//2, SCREEN_HEIGHT//2 - 50))
        pygame.display.flip()
        
        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_o:
                        self.wearing_glasses = True
                        waiting = False
                    elif event.key == pygame.K_n:
                        self.wearing_glasses = False
                        waiting = False
                elif event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
    
    def load_or_generate_test_data(self):
        """Charge ou génère des données de test"""
        # Texte pour les tests de préférence
        self.preference_pairs = [
            (("b", "d"), ("ʙ", "ð"), "Quelle forme distinguez-vous le mieux?"),
            (("p", "q"), ("þ", "Ɋ"), "Laquelle de ces lettres est plus lisible?"),
            (("a", "e"), ("ɑ", "ɛ"), "Préférez-vous la gauche ou la droite?"),
            (("m", "n"), ("ᴍ", "ɴ"), "Quelle version vous semble plus claire?"),
            (("u", "v"), ("ᴜ", "ᴠ"), "Laquelle identifiez-vous plus facilement?")
        ]
        
        # Texte pour l'analyse linguistique
        self.linguistic_text = (
            "La dyslexie est un trouble spécifique de l'apprentissage du langage écrit. "
            "Elle se caractérise par des difficultés dans l'acquisition et dans l'automatisation "
            "des mécanismes nécessaires à la maîtrise de l'écrit. Ces difficultés résultent "
            "d'un déficit au niveau des processus cognitifs fondamentaux. "
            "Les personnes dyslexiques peuvent confondre des lettres comme 'b' et 'd', "
            "'p' et 'q', ou inverser des syllabes. Ces confusions rendent la lecture "
            "laborieuse et peuvent entraîner des erreurs d'orthographe persistantes "
            "malgré un enseignement adapté."
        )
        
        # Définir les transformations possibles avec plus de variantes
        self.letter_variants = {
            'a': ['a', 'ɑ', 'ɐ'],
            'b': ['b', 'ʙ', 'ɓ'],
            'c': ['c', 'ꞓ', 'ↄ'],
            'd': ['d', 'ð', 'ɖ'],
            'e': ['e', 'ɛ', 'ɘ'],
            'f': ['f', 'ꝼ', 'ʄ'],
            'g': ['g', 'ɡ', 'ɠ'],
            'h': ['h', 'ɦ', 'ʜ'],
            'i': ['i', 'ɨ', 'ɪ'],
            'j': ['j', 'ʝ', 'ɟ'],
            'k': ['k', 'ꝁ', 'ʞ'],
            'l': ['l', 'ʟ', 'ɭ'],
            'm': ['m', 'ᴍ', 'ɱ'],
            'n': ['n', 'ɴ', 'ɲ'],
            'o': ['o', 'ɔ', 'ɵ'],
            'p': ['p', 'þ', 'ƿ'],
            'q': ['q', 'Ɋ', 'ʠ'],
            'r': ['r', 'ɹ', 'ʀ'],
            's': ['s', 'ʂ', 'ꞩ'],
            't': ['t', 'ʈ', 'ŧ'],
            'u': ['u', 'ᴜ', 'ʉ'],
            'v': ['v', 'ᴠ', 'ʋ'],
            'w': ['w', 'ᴡ', 'ʍ'],
            'x': ['x', 'ꭓ', 'ˣ'],
            'y': ['y', 'ʏ', 'ɏ'],
            'z': ['z', 'ʐ', 'ƶ']
        }
        
        # Charger le profil utilisateur s'il existe
        if os.path.exists(USER_PROFILE_PATH):
            with open(USER_PROFILE_PATH, 'r') as f:
                self.user_profile = json.load(f)
                print("Profil utilisateur chargé")
    
    def save_user_profile(self):
        """Sauvegarde le profil utilisateur"""
        with open(USER_PROFILE_PATH, 'w') as f:
            json.dump(self.user_profile, f)
        print("Profil utilisateur sauvegardé")
    
    def run_preference_test(self):
        """Effectue un test de préférence avec l'utilisateur"""
        print("Début du test de préférence...")
        
        for i, ((orig1, orig2), (var1, var2), question_text) in enumerate(self.preference_pairs):
            if not self.running:
                break
                
            self.screen.fill((30, 30, 50))
            
            # Afficher la question
            question = self.font.render(question_text, True, (255, 255, 255))
            self.screen.blit(question, (SCREEN_WIDTH//2 - question.get_width()//2, SCREEN_HEIGHT//4))
            
            # Afficher les options
            option1 = self.font.render(f"{orig1} vs {var1}", True, (255, 255, 0))
            option2 = self.font.render(f"{orig2} vs {var2}", True, (0, 255, 255))
            
            self.screen.blit(option1, (SCREEN_WIDTH//4 - option1.get_width()//2, SCREEN_HEIGHT//2))
            self.screen.blit(option2, (3*SCREEN_WIDTH//4 - option2.get_width()//2, SCREEN_HEIGHT//2))
            
            instruction = self.small_font.render("Appuyez sur 1 ou 2 pour choisir", True, (200, 200, 200))
            self.screen.blit(instruction, (SCREEN_WIDTH//2 - instruction.get_width()//2, SCREEN_HEIGHT - 100))
            
            pygame.display.flip()
            
            # Enregistrer les données oculaires pendant la décision
            gaze_data = []
            start_time = time.time()
            decision_time = 0
            choice = None
            
            while choice is None and self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Détecter les yeux et estimer le regard
                left_eye, right_eye = self.eye_tracker.detect_eyes(frame)
                if left_eye and right_eye:
                    frame_height, frame_width = frame.shape[:2]
                    left_center, left_angle, left_dist = self.eye_tracker.estimate_gaze(left_eye, frame_width, frame_height)
                    right_center, right_angle, right_dist = self.eye_tracker.estimate_gaze(right_eye, frame_width, frame_height)
                    
                    avg_angle = (left_angle + right_angle) / 2
                    avg_dist = (left_dist + right_dist) / 2
                    
                    # Calculer la durée depuis le dernier échantillon
                    current_time = time.time()
                    duration = current_time - self.last_fixation_time
                    self.last_fixation_time = current_time
                    
                    gaze_data.append({
                        'timestamp': current_time - start_time,
                        'angle': avg_angle,
                        'distance': avg_dist,
                        'duration': duration
                    })
                
                # Vérifier l'entrée utilisateur
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_1:
                            choice = 1
                            decision_time = time.time() - start_time
                        elif event.key == pygame.K_2:
                            choice = 2
                            decision_time = time.time() - start_time
                    elif event.type == pygame.QUIT:
                        self.running = False
            
            if choice is None:
                break
                
            # Enregistrer les résultats du test
            self.preference_test_data.append({
                'pair': (orig1, orig2),
                'variants': (var1, var2),
                'choice': choice,
                'decision_time': decision_time,
                'gaze_data': gaze_data
            })
            
            # Mettre à jour le profil utilisateur
            if choice == 1:
                self.user_profile[orig1] = var1
            else:
                self.user_profile[orig2] = var2
            
            # Sauvegarder après chaque choix
            self.save_user_profile()
        
        print("Test de préférence terminé.")
    
    def apply_transformation(self, text):
        """Applique la transformation au texte en fonction du modèle et du profil"""
        # Si nous avons des préférences utilisateur, les utiliser en priorité
        transformed_text = ""
        for char in text:
            if char.lower() in self.user_profile:
                transformed_text += self.user_profile[char.lower()].upper() if char.isupper() else self.user_profile[char.lower()]
            else:
                transformed_text += char
        
        return transformed_text
    
    def run_linguistic_analysis(self):
        """Exécute l'analyse linguistique avec suivi oculaire"""
        print("Début de l'analyse linguistique...")
        
        # Appliquer une transformation basée sur le profil
        transformed_text = self.apply_transformation(self.linguistic_text)
        
        # Afficher le texte
        self.display_text(transformed_text)
        
        # Collecter des données pendant la lecture
        reading_data = []
        last_update_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Détecter les yeux et estimer le regard
            left_eye, right_eye = self.eye_tracker.detect_eyes(frame)
            if left_eye and right_eye:
                frame_height, frame_width = frame.shape[:2]
                left_center, left_angle, left_dist = self.eye_tracker.estimate_gaze(left_eye, frame_width, frame_height)
                right_center, right_angle, right_dist = self.eye_tracker.estimate_gaze(right_eye, frame_width, frame_height)
                
                avg_angle = (left_angle + right_angle) / 2
                avg_dist = (left_dist + right_dist) / 2
                
                # Calculer la durée de fixation
                current_time = time.time()
                duration = current_time - self.last_fixation_time
                self.last_fixation_time = current_time
                
                # Mettre à jour l'historique du regard
                self.gaze_history.append((avg_angle, avg_dist, duration, current_time))
                
                # Enregistrer les données pour l'analyse
                reading_data.append({
                    'timestamp': current_time,
                    'angle': avg_angle,
                    'distance': avg_dist,
                    'duration': duration
                })
                
                # Afficher les données de suivi oculaire
                self.display_gaze_data(avg_angle, avg_dist, duration)
            
            # Mettre à jour la transformation périodiquement
            if time.time() - last_update_time > 2.0:  # Toutes les 2 secondes
                transformed_text = self.apply_transformation(self.linguistic_text)
                self.display_text(transformed_text)
                last_update_time = time.time()
            
            # Vérifier les événements
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    # Forcer une mise à jour
                    transformed_text = self.apply_transformation(self.linguistic_text)
                    self.display_text(transformed_text)
                    last_update_time = time.time()
                elif event.type == pygame.QUIT:
                    self.running = False
        
        print("Analyse linguistique terminée.")
        return reading_data
    
    def display_text(self, text):
        """Affiche le texte à l'écran avec mise en forme améliorée"""
        self.screen.fill((30, 30, 50))
        
        # Titre
        title = self.title_font.render("Texte Adapté", True, (0, 200, 255))
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 20))
        
        # Diviser le texte en lignes
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + word + " "
            text_width = self.font.size(test_line)[0]
            
            if text_width < SCREEN_WIDTH - 2 * TEXT_MARGIN:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word + " "
        
        if current_line:
            lines.append(current_line)
        
        # Afficher les lignes avec espacement
        y_pos = TEXT_MARGIN + 70
        line_spacing = FONT_SIZE + 10
        
        for i, line in enumerate(lines):
            # Mettre en évidence la ligne actuelle basée sur le regard
            if len(self.gaze_history) > 0:
                gaze_y = self.eye_tracker.map_to_screen(self.gaze_history[-1][:2])[1]
                if TEXT_MARGIN + 70 + i * line_spacing < gaze_y < TEXT_MARGIN + 70 + (i+1) * line_spacing:
                    pygame.draw.rect(self.screen, (50, 50, 80), 
                                    (TEXT_MARGIN - 10, y_pos - 5, 
                                     SCREEN_WIDTH - 2 * TEXT_MARGIN + 20, line_spacing))
            
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (TEXT_MARGIN, y_pos))
            y_pos += line_spacing
        
        pygame.display.flip()
    
    def display_gaze_data(self, angle, distance, duration):
        """Affiche les données de suivi oculaire avec une interface améliorée"""
        # Afficher l'angle
        angle_text = self.small_font.render(f"Angle: {math.degrees(angle):.1f}°", True, (0, 255, 0))
        self.screen.blit(angle_text, (SCREEN_WIDTH - 250, 20))
        
        # Afficher la distance
        dist_text = self.small_font.render(f"Distance: {distance:.2f}", True, (0, 255, 0))
        self.screen.blit(dist_text, (SCREEN_WIDTH - 250, 50))
        
        # Afficher la durée de fixation
        dur_text = self.small_font.render(f"Durée fixation: {duration:.2f}s", True, (0, 255, 0))
        self.screen.blit(dur_text, (SCREEN_WIDTH - 250, 80))
        
        # Afficher le port de lunettes
        glasses_text = self.small_font.render(
            f"Lunettes: {'Oui' if self.wearing_glasses else 'Non'}", 
            True, (0, 255, 255)
        )
        self.screen.blit(glasses_text, (SCREEN_WIDTH - 250, 110))
        
        # Afficher les statistiques de Maxwell
        if self.fixation_data:
            durations = [f['duration'] for f in self.fixation_data]
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            maxwell_param = (2 * std_duration**2) / (np.pi - 2)  # Approximation
            
            stats_text = self.small_font.render(
                f"Durée moyenne: {mean_duration:.2f}s", 
                True, (255, 255, 0)
            )
            self.screen.blit(stats_text, (SCREEN_WIDTH - 400, 140))
            
            maxwell_text = self.small_font.render(
                f"Paramètre Maxwell: {maxwell_param:.2f}", 
                True, (255, 255, 0)
            )
            self.screen.blit(maxwell_text, (SCREEN_WIDTH - 400, 170))
        
        # Afficher le point de regard estimé
        if self.gaze_history:
            gaze_vector = (angle, distance)
            screen_point = self.eye_tracker.map_to_screen(gaze_vector)
            pygame.draw.circle(self.screen, (255, 0, 0), 
                             (int(screen_point[0]), int(screen_point[1])), 
                             8, 2)
        
        pygame.display.flip()
    
    def run(self):
        """Exécute le système principal"""
        if not self.calibrated:
            print("Le système n'est pas calibré. Arrêt.")
            return
        
        # Entraîner le modèle si nécessaire
        if not os.path.exists(MODEL_PATH):
            print("Entraînement initial du modèle...")
            self.train_model(epochs=10)
        
        # Exécuter le test de préférence
        self.run_preference_test()
        
        # Exécuter l'analyse linguistique
        self.fixation_data = self.run_linguistic_analysis()
        
        # Nettoyage
        self.cleanup()
    
    def cleanup(self):
        """Nettoyage des ressources"""
        self.cap.release()
        if hasattr(self.eye_tracker, 'face_mesh'):
            self.eye_tracker.face_mesh.close()
        pygame.quit()
        print("Système arrêté proprement")

# Point d'entrée principal
if __name__ == "__main__":
    try:
        system = DyslexicAdaptationSystem()
        system.run()
    except Exception as e:
        print(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
