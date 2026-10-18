import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import cv2
import os
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib.font_manager as fm
import random
import logging
from scipy.ndimage import gaussian_filter
from collections import defaultdict
import math

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dyslexia_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Configuration avancée
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FONT_SIZES = [80, 100, 120]  # Taille de police variable
IMAGE_SIZE = 256  # Résolution plus élevée
MODEL_PATH = "holomorphic_profile.json"
BACKGROUND_COLORS = [(0, 0, 0), (25, 25, 25), (50, 50, 50)]  # Fond sombre pour réduire la fatigue oculaire
TEXT_COLORS = [(255, 255, 255), (220, 220, 255), (255, 220, 220)]  # Couleurs de texte adaptées

# Classe pour la transformation holomorphique avancée
class AdvancedHolomorphicTransformer:
    def _init_(self):
        # Paramètres par défaut avec plages étendues
        self.default_params = {
            'rotation': 0.0,
            'scale': 1.0,
            'translation_re': 0.0,
            'translation_im': 0.0,
            'shear_re': 0.0,
            'shear_im': 0.0,
            'curvature': 0.0,
            'complex_phase': 0.0,
            'complex_magnitude': 1.0,
            'thickness': 1.0,  # Épaisseur du trait
            'smoothing': 0.1,  # Lissage des contours
            'asymmetry': 0.0   # Asymétrie contrôlée
        }
        
        # Polices multiples pour variété
        self.fonts = self.load_fonts()
        self.current_font_idx = 0
        
        # Cache pour les images de lettres
        self.letter_cache = {}
        
        # Statistiques d'utilisation
        self.usage_stats = defaultdict(int)
    
    def load_fonts(self):
        """Charge plusieurs polices adaptées pour dyslexie"""
        fonts = []
        font_families = [
            "OpenDyslexic", "Comic Sans MS", "Arial", "Verdana", 
            "Gill Sans", "Trebuchet MS", "Calibri"
        ]
        
        for family in font_families:
            try:
                font_path = fm.findfont(fm.FontProperties(family=family, weight="normal"))
                fonts.append(ImageFont.truetype(font_path, max(FONT_SIZES)))
                logger.info(f"Police chargée: {font_path}")
            except:
                logger.warning(f"Police non trouvée: {family}")
        
        if not fonts:
            logger.warning("Aucune police trouvée, utilisation de la police par défaut")
            default_font = ImageFont.load_default()
            default_font.size = max(FONT_SIZES)
            fonts.append(default_font)
        
        return fonts
    
    def get_font(self, size_idx=1):
        """Retourne une police avec taille adaptée"""
        size = FONT_SIZES[size_idx]
        font = self.fonts[self.current_font_idx]
        try:
            # Crée une nouvelle instance avec la taille désirée
            return ImageFont.truetype(font.path, size)
        except:
            # Fallback si la modification de taille échoue
            return font
    
    def generate_letter_image(self, letter, bg_color_idx=0, text_color_idx=0):
        """Génère une image de la lettre avec anti-aliasing et fond personnalisé"""
        cache_key = f"{letter}{bg_color_idx}{text_color_idx}"
        if cache_key in self.letter_cache:
            return self.letter_cache[cache_key]
        
        # Créer une image avec fond coloré
        bg_color = BACKGROUND_COLORS[bg_color_idx]
        text_color = TEXT_COLORS[text_color_idx]
        
        # Utiliser une image temporaire plus grande pour l'anti-aliasing
        temp_size = IMAGE_SIZE * 2
        img = Image.new('RGB', (temp_size, temp_size), bg_color)
        draw = ImageDraw.Draw(img)
        
        try:
            # Taille de police adaptative
            font_size_idx = min(2, max(0, len(letter) - 1))  # Plus grande pour les caractères uniques
            font = self.get_font(font_size_idx)
            
            # Centrage précis
            bbox = draw.textbbox((0, 0), letter, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (temp_size - text_width) / 2 - bbox[0]
            y = (temp_size - text_height) / 2 - bbox[1]
            
            # Dessiner avec anti-aliasing
            draw.text((x, y), letter, fill=text_color, font=font)
            
            # Réduire avec anti-aliasing
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            img_array = np.array(img) / 255.0
            
            # Mettre en cache
            self.letter_cache[cache_key] = img_array
            self.usage_stats[letter] += 1
            
            return img_array
        except Exception as e:
            logger.error(f"Erreur génération lettre {letter}: {e}")
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

    def get_letter_contour(self, letter_image, simplify=True):
        """Extrait le contour avec plusieurs techniques"""
        try:
            # Convertir en niveaux de gris pour le traitement
            gray = cv2.cvtColor((letter_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Seuillage adaptatif
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Trouver les contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return np.array([[-0.9, -0.9], [-0.9, 0.9], [0.9, 0.9], [0.9, -0.9]])
            
            # Fusionner les contours proches
            main_contour = np.vstack(contours)
            
            if simplify and len(main_contour) > 50:
                # Simplification avec l'algorithme de Douglas-Peucker
                epsilon = 0.005 * cv2.arcLength(main_contour, True)
                approx = cv2.approxPolyDP(main_contour, epsilon, True)
                points = approx.squeeze(1).astype(np.float32)
            else:
                points = main_contour.squeeze(1).astype(np.float32)
            
            # Normalisation
            points[:, 0] = 2 * (points[:, 0] / IMAGE_SIZE) - 1
            points[:, 1] = 2 * (points[:, 1] / IMAGE_SIZE) - 1
            
            # Centrer les points
            centroid = np.mean(points, axis=0)
            points -= centroid
            
            return points
        except Exception as e:
            logger.error(f"Erreur d'extraction de contour: {e}")
            return np.array([[-0.9, -0.9], [-0.9, 0.9], [0.9, 0.9], [0.9, -0.9]])

    def holomorphic_transform(self, points, params):
        """Transformation holomorphique avancée avec plusieurs effets combinés"""
        try:
            # Convertir en coordonnées complexes
            z = points[:, 0] + 1j * points[:, 1]
            
            # Paramètres de transformation
            rotation = params.get('rotation', 0)
            scale = params.get('scale', 1)
            translation = params.get('translation_re', 0) + 1j * params.get('translation_im', 0)
            shear = params.get('shear_re', 0) + 1j * params.get('shear_im', 0)
            curvature = params.get('curvature', 0)
            phase = params.get('complex_phase', 0)
            magnitude = params.get('complex_magnitude', 1)
            thickness = params.get('thickness', 1)
            asymmetry = params.get('asymmetry', 0)
            
            # Transformation de base
            w = magnitude * np.exp(1j * phase) * z + translation + shear * np.conj(z)
            
            # Rotation et mise à l'échelle
            w = scale * np.exp(1j * rotation) * w
            
            # Courbure holomorphique
            w += curvature * np.exp(1j * np.angle(w))
            
            # Effet d'épaisseur (dilatation/érosion)
            w *= thickness
            
            # Asymétrie contrôlée
            if asymmetry != 0:
                asymmetry_factor = 1 + asymmetry * np.real(w)
                w *= asymmetry_factor
            
            # Convertir en coordonnées cartésiennes
            return np.column_stack((np.real(w), np.imag(w)))
        except Exception as e:
            logger.error(f"Erreur de transformation: {e}")
            return points

    def apply_holomorphic_transform(self, points, params):
        """Applique la transformation avec interpolation et lissage"""
        try:
            transformed_points = self.holomorphic_transform(points, params)
            
            if len(transformed_points) > 3:
                # Paramètre de lissage basé sur la complexité
                smoothness = max(0.001, min(0.1, params.get('smoothing', 0.1)))
                tck, u = splprep(transformed_points.T, s=smoothness * len(transformed_points))
                u_new = np.linspace(u.min(), u.max(), max(50, len(points)))
                interpolated = np.column_stack(splev(u_new, tck))
                return interpolated
            return transformed_points
        except:
            return points

    def transform_letter_image(self, transformed_points, bg_color_idx=0, text_color_idx=0):
        """Crée une nouvelle image avec effets visuels avancés"""
        try:
            points = transformed_points.copy()
            
            # Dénormalisation
            points[:, 0] = (points[:, 0] + 1) * IMAGE_SIZE / 2
            points[:, 1] = (points[:, 1] + 1) * IMAGE_SIZE / 2
            
            # Créer une image avec fond coloré
            bg_color = BACKGROUND_COLORS[bg_color_idx]
            text_color = TEXT_COLORS[text_color_idx]
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), bg_color)
            draw = ImageDraw.Draw(img)
            
            if len(points) > 2:
                # Dessiner le contour rempli
                draw.polygon([(x, y) for x, y in points], fill=text_color)
                
                # Ajouter un léger flou pour l'anti-aliasing
                img_array = np.array(img)
                img_array = gaussian_filter(img_array, sigma=0.7)
                img = Image.fromarray(np.uint8(img_array))
            
            return np.array(img) / 255.0
        except:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

    def generate_transformation_params(self, base_params=None):
        """Génère des paramètres aléatoires avec cohérence"""
        params = self.default_params.copy()
        if base_params:
            params.update(base_params)
        
        # Modifications cohérentes
        rotation_var = random.uniform(-0.4, 0.4)
        params['rotation'] += rotation_var
        params['scale'] *= random.uniform(0.9, 1.1)
        params['translation_re'] += random.uniform(-0.1, 0.1)
        params['translation_im'] += random.uniform(-0.1, 0.1)
        params['shear_re'] += random.uniform(-0.2, 0.2)
        params['shear_im'] += random.uniform(-0.2, 0.2)
        params['curvature'] += random.uniform(-0.3, 0.3)
        params['complex_phase'] += random.uniform(-1.0, 1.0)
        params['complex_magnitude'] *= random.uniform(0.9, 1.1)
        params['thickness'] = random.uniform(0.8, 1.2)
        params['smoothing'] = random.uniform(0.05, 0.2)
        params['asymmetry'] = random.uniform(-0.1, 0.1)
        
        return params

    def transform_letter(self, letter, params=None, bg_color_idx=0, text_color_idx=0):
        """Transforme une lettre avec paramètres personnalisés"""
        original_image = self.generate_letter_image(letter, bg_color_idx, text_color_idx)
        contour = self.get_letter_contour(original_image)
        
        if params is None:
            params = self.generate_transformation_params()
        
        transformed_contour = self.apply_holomorphic_transform(contour, params)
        transformed_image = self.transform_letter_image(transformed_contour, bg_color_idx, text_color_idx)
        
        return original_image, transformed_image, params

    def save_profile(self, profile):
        """Sauvegarde le profil complet"""
        profile['usage_stats'] = dict(self.usage_stats)
        with open(MODEL_PATH, 'w') as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Profil sauvegardé: {MODEL_PATH}")

    def load_profile(self):
        """Charge le profil complet"""
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'r') as f:
                profile = json.load(f)
            logger.info(f"Profil chargé: {MODEL_PATH}")
            
            # Mettre à jour les statistiques
            self.usage_stats.update(profile.get('usage_stats', {}))
            return profile
        return None

# Classe pour l'interface utilisateur avancée
class DyslexiaAssistant:
    def _init_(self):
        self.transformer = AdvancedHolomorphicTransformer()
        self.user_profile = {
            'preferences': self.transformer.default_params.copy(),
            'letter_mapping': {},
            'color_prefs': {'background': 0, 'text': 0},
            'font_prefs': {'current_idx': 0, 'sizes': {}},
            'dyslexia_type': None
        }
        
        # Charger le profil existant
        saved_profile = self.transformer.load_profile()
        if saved_profile:
            self.user_profile.update(saved_profile)
            self.transformer.current_font_idx = self.user_profile['font_prefs'].get('current_idx', 0)
        
        # Configuration UI
        self.current_text = "Bienvenue dans le système d'adaptation pour dyslexie"
        self.preview_mode = True
    
    def run_dyslexia_analysis(self):
        """Analyse complète de la dyslexie avec typage"""
        print("\n" + "="*60)
        print("ANALYSE DE LA PERCEPTION DYSLEXIQUE".center(60))
        print("="*60)
        
        # Questionnaire détaillé
        self.user_profile['dyslexia_type'] = self.identify_dyslexia_type()
        
        # Test de perception
        print("\n[PHASE 1: TEST DES LETTRES CONFONDUES]")
        self.test_letter_confusions()
        
        # Test de sensibilité aux couleurs
        print("\n[PHASE 2: TEST DE SENSIBILITÉ AUX COULEURS]")
        self.test_color_sensitivity()
        
        # Test de préférence de police
        print("\n[PHASE 3: TEST DE PRÉFÉRENCE DE POLICE]")
        self.test_font_preference()
        
        # Générer le profil final
        self.generate_personalized_profile()
        
        print("\n" + "="*60)
        print("ANALYSE TERMINÉE AVEC SUCCÈS".center(60))
        print("="*60)
    
    def identify_dyslexia_type(self):
        """Identifie le type de dyslexie"""
        print("\nVeuillez répondre aux questions suivantes :")
        score = 0
        
        questions = [
            ("Avez-vous souvent des difficultés à distinguer 'b' et 'd', 'p' et 'q'?", 2),
            ("Les lettres semblent-elles parfois bouger ou se chevaucher?", 3),
            ("Avez-vous des difficultés à suivre les lignes de texte?", 2),
            ("La lecture vous fatigue-t-elle rapidement les yeux?", 1),
            ("Préférez-vous les polices sans empattement comme Arial ou Comic Sans?", 1),
            ("Les textes sur fond coloré sont-ils plus faciles à lire?", 1),
            ("Avez-vous des difficultés avec l'orthographe malgré une bonne compréhension orale?", 2)
        ]
        
        for q, weight in questions:
            while True:
                response = input(f"\n{q} (o/n) ").strip().lower()
                if response in ['o', 'n']:
                    if response == 'o':
                        score += weight
                    break
                print("Veuillez répondre par 'o' (oui) ou 'n' (non)")
        
        # Déterminer le type de dyslexie
        if score >= 10:
            dyslexia_type = "Dyslexie visuo-attentionnelle sévère"
        elif score >= 7:
            dyslexia_type = "Dyslexie phonologique avec composante visuelle"
        elif score >= 4:
            dyslexia_type = "Dyslexie légère de surface"
        else:
            dyslexia_type = "Profil neurotypique avec sensibilités spécifiques"
        
        print(f"\nType identifié: {dyslexia_type}")
        return dyslexia_type
    
    def test_letter_confusions(self):
        """Test complet des confusions de lettres"""
        letter_pairs = [
            ('b', 'd'), ('p', 'q'), ('a', 'e'), ('n', 'm'), ('u', 'v'),
            ('f', 't'), ('c', 'o'), ('s', 'z'), ('r', 'n'), ('h', 'k'),
            ('m', 'w'), ('i', 'j'), ('v', 'y'), ('5', 's'), ('6', 'b')
        ]
        
        print("\nPour chaque paire, choisissez l'option la plus lisible")
        print("1: Première lettre, 2: Deuxième lettre, 3: Aucune préférence")
        
        for l1, l2 in letter_pairs:
            while True:
                # Afficher les options visuelles
                self.display_letter_comparison(l1, l2)
                
                choice = input(f"\n{l1} vs {l2}? (1/2/3) ").strip()
                if choice in ['1', '2', '3']:
                    if choice == '1':
                        self.user_profile['letter_mapping'][l2] = l1
                    elif choice == '2':
                        self.user_profile['letter_mapping'][l1] = l2
                    break
                print("Choix invalide. Veuillez entrer 1, 2 ou 3.")
    
    def test_color_sensitivity(self):
        """Test de sensibilité aux combinaisons de couleurs"""
        print("\nQuelle combinaison est la plus confortable pour la lecture?")
        
        combinations = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2)
        ]
        
        # Afficher toutes les combinaisons
        self.display_color_matrix(combinations)
        
        while True:
            choice = input("\nEntrez le numéro de la meilleure combinaison (1-9): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= 9:
                bg_idx, text_idx = combinations[int(choice)-1]
                self.user_profile['color_prefs']['background'] = bg_idx
                self.user_profile['color_prefs']['text'] = text_idx
                break
            print("Choix invalide. Veuillez entrer un nombre entre 1 et 9.")
    
    def test_font_preference(self):
        """Test de préférence de police"""
        print("\nQuelle police est la plus lisible pour vous?")
        
        # Afficher toutes les polices
        self.display_font_options()
        
        while True:
            choice = input("\nEntrez le numéro de la police préférée (1-{}): ".format(len(self.transformer.fonts))).strip()
            if choice.isdigit() and 1 <= int(choice) <= len(self.transformer.fonts):
                font_idx = int(choice) - 1
                self.transformer.current_font_idx = font_idx
                self.user_profile['font_prefs']['current_idx'] = font_idx
                break
            print("Choix invalide. Veuillez entrer un numéro valide.")
    
    def generate_personalized_profile(self):
        """Génère un profil personnalisé basé sur les résultats"""
        # Ajustements basés sur le type de dyslexie
        if "visuo-attentionnelle" in self.user_profile['dyslexia_type']:
            self.user_profile['preferences']['thickness'] = 1.2
            self.user_profile['preferences']['scale'] = 1.15
            self.user_profile['preferences']['asymmetry'] = -0.05
        elif "phonologique" in self.user_profile['dyslexia_type']:
            self.user_profile['preferences']['curvature'] = 0.1
            self.user_profile['preferences']['complex_phase'] = 0.2
        else:
            self.user_profile['preferences']['smoothing'] = 0.15
        
        # Ajustements basés sur les confusions de lettres
        b_count = sum(1 for k in self.user_profile['letter_mapping'].values() if k == 'b')
        d_count = sum(1 for k in self.user_profile['letter_mapping'].values() if k == 'd')
        
        if b_count > d_count:
            self.user_profile['preferences']['rotation'] = -0.1
        else:
            self.user_profile['preferences']['rotation'] = 0.1
        
        # Sauvegarder le profil
        self.transformer.save_profile(self.user_profile)
        print("\nProfil personnel généré et sauvegardé")
    
    def transform_text(self, text):
        """Transforme un texte entier selon le profil"""
        original_images = []
        transformed_images = []
        transformed_text = ""
        
        bg_idx = self.user_profile['color_prefs']['background']
        text_idx = self.user_profile['color_prefs']['text']
        
        for char in text:
            if char in LETTERS:
                # Appliquer le mapping
                mapped_char = self.user_profile['letter_mapping'].get(char, char)
                
                # Générer les images
                orig_img, trans_img, _ = self.transformer.transform_letter(
                    mapped_char, 
                    self.user_profile['preferences'],
                    bg_idx,
                    text_idx
                )
                original_images.append(orig_img)
                transformed_images.append(trans_img)
                transformed_text += mapped_char
            else:
                # Pour les espaces et caractères spéciaux
                blank = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
                original_images.append(blank)
                transformed_images.append(blank)
                transformed_text += char
        
        return original_images, transformed_images, transformed_text
    
    def display_letter_comparison(self, l1, l2):
        """Affiche une comparaison visuelle de deux lettres"""
        bg_idx = self.user_profile['color_prefs']['background']
        text_idx = self.user_profile['color_prefs']['text']
        
        _, img1, _ = self.transformer.transform_letter(l1, bg_color_idx=bg_idx, text_color_idx=text_idx)
        _, img2, _ = self.transformer.transform_letter(l2, bg_color_idx=bg_idx, text_color_idx=text_idx)
        
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.imshow(img1)
        plt.title(f'Option 1: {l1}')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(img2)
        plt.title(f'Option 2: {l2}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"comparison_{l1}_{l2}.png")
        plt.show(block=False)
        plt.pause(0.1)
    
    def display_color_matrix(self, combinations):
        """Affiche une matrice de combinaisons de couleurs"""
        plt.figure(figsize=(12, 8))
        plt.suptitle("Sélectionnez la combinaison la plus confortable", fontsize=16)
        
        for i, (bg_idx, text_idx) in enumerate(combinations):
            plt.subplot(3, 3, i+1)
            
            # Créer un échantillon de texte
            img = np.zeros((100, 100, 3))
            img[:, :] = np.array(BACKGROUND_COLORS[bg_idx]) / 255.0
            
            # Ajouter du texte
            text_color = np.array(TEXT_COLORS[text_idx]) / 255.0
            cv2.putText(
                img, "Texte", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                text_color, 2, cv2.LINE_AA
            )
            
            plt.imshow(img)
            plt.title(f"Option {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("color_options.png")
        plt.show(block=False)
        plt.pause(0.1)
    
    def display_font_options(self):
        """Affiche toutes les options de police"""
        plt.figure(figsize=(12, 8))
        plt.suptitle("Sélectionnez la police la plus lisible", fontsize=16)
        
        bg_idx = self.user_profile['color_prefs']['background']
        text_idx = self.user_profile['color_prefs']['text']
        
        for i in range(len(self.transformer.fonts)):
            plt.subplot(3, 3, i+1)
            
            # Temporairement changer de police
            original_font_idx = self.transformer.current_font_idx
            self.transformer.current_font_idx = i
            
            # Générer un échantillon
            sample_text = "Sample"
            _, img, _ = self.transformer.transform_letter(
                sample_text, 
                bg_color_idx=bg_idx,
                text_color_idx=text_idx
            )
            
            plt.imshow(img)
            plt.title(f"Option {i+1}")
            plt.axis('off')
            
            # Restaurer la police originale
            self.transformer.current_font_idx = original_font_idx
        
        plt.tight_layout()
        plt.savefig("font_options.png")
        plt.show(block=False)
        plt.pause(0.1)
    
    def create_text_document(self, text, filename="document_adapté.pdf"):
        """Crée un document PDF avec le texte transformé"""
        try:
            from fpdf import FPDF
            from PIL import Image as PILImage
            import tempfile
            
            bg_color = BACKGROUND_COLORS[self.user_profile['color_prefs']['background']]
            text_color = TEXT_COLORS[self.user_profile['color_prefs']['text']]
            
            # Créer un PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Définir les couleurs
            pdf.set_fill_color(*bg_color)
            pdf.set_text_color(*text_color)
            
            # Utiliser une police adaptée si disponible
            try:
                font_path = self.transformer.fonts[self.transformer.current_font_idx].path
                pdf.add_font('DyslexicFont', '', font_path, uni=True)
                pdf.set_font('DyslexicFont', '', 14)
            except:
                pdf.set_font('Arial', '', 14)
            
            # Ajouter le texte transformé
            pdf.multi_cell(0, 10, text)
            
            # Enregistrer le PDF
            pdf.output(filename)
            logger.info(f"Document généré: {filename}")
            return True
        except ImportError:
            logger.warning("fpdf non installé, impossible de générer un PDF")
            return False
        except Exception as e:
            logger.error(f"Erreur génération PDF: {e}")
            return False
    
    def run(self):
        """Interface principale de l'assistant"""
        print("\n" + "="*60)
        print("ASSISTANT D'ADAPTATION POUR DYSLEXIE".center(60))
        print("="*60)
        
        # Analyse initiale
        if not self.user_profile.get('dyslexia_type'):
            print("\nAucun profil détecté, démarrage de l'analyse...")
            self.run_dyslexia_analysis()
        else:
            print(f"\nProfil chargé: {self.user_profile['dyslexia_type']}")
        
        # Menu principal
        while True:
            print("\n" + "="*30)
            print("MENU PRINCIPAL".center(30))
            print("="*30)
            print("1. Visualiser une lettre transformée")
            print("2. Adapter un texte personnalisé")
            print("3. Générer un document PDF adapté")
            print("4. Réexécuter l'analyse de dyslexie")
            print("5. Afficher les statistiques d'usage")
            print("6. Quitter")
            
            choice = input("\nVotre choix: ").strip()
            
            if choice == '1':
                letter = input("Entrez une lettre: ").strip()[0]
                self.display_letter_transformation(letter)
            elif choice == '2':
                self.process_custom_text()
            elif choice == '3':
                text = input("Entrez le texte à convertir en PDF: ")
                self.create_text_document(text)
            elif choice == '4':
                self.run_dyslexia_analysis()
            elif choice == '5':
                self.display_usage_stats()
            elif choice == '6':
                print("\nMerci d'avoir utilisé l'assistant d'adaptation pour dyslexie")
                break
            else:
                print("Choix invalide, veuillez réessayer.")
    
    def display_letter_transformation(self, letter):
        """Affiche la transformation d'une lettre"""
        bg_idx = self.user_profile['color_prefs']['background']
        text_idx = self.user_profile['color_prefs']['text']
        
        orig_img, trans_img, _ = self.transformer.transform_letter(
            letter, 
            self.user_profile['preferences'],
            bg_idx,
            text_idx
        )
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(orig_img)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(trans_img)
        plt.title('Adapté')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"letter_{letter}.png")
        plt.show()
    
    def process_custom_text(self):
        """Traite et affiche un texte personnalisé"""
        text = input("\nEntrez le texte à adapter: ")
        orig_imgs, trans_imgs, trans_text = self.transform_text(text)
        
        # Afficher la comparaison
        self.display_text_comparison(orig_imgs, trans_imgs, text, trans_text)
        
        print("\nTexte transformé:")
        print(trans_text)
    
    def display_text_comparison(self, orig_imgs, trans_imgs, orig_text, trans_text):
        """Affiche une comparaison côte à côte des textes"""
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        
        # Texte original
        axs[0].set_title("Texte Original")
        x_offset = 0
        for img in orig_imgs:
            axs[0].imshow(img, extent=[x_offset, x_offset+1, 0, 1])
            x_offset += 1
        axs[0].set_xlim(0, len(orig_text))
        axs[0].set_ylim(0, 1)
        axs[0].axis('off')
        
        # Texte transformé
        axs[1].set_title("Texte Adapté")
        x_offset = 0
        for img in trans_imgs:
            axs[1].imshow(img, extent=[x_offset, x_offset+1, 0, 1])
            x_offset += 1
        axs[1].set_xlim(0, len(trans_text))
        axs[1].set_ylim(0, 1)
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("text_comparison.png")
        plt.show()
    
    def display_usage_stats(self):
        """Affiche les statistiques d'usage"""
        print("\nSTATISTIQUES D'USAGE:")
        total = sum(self.transformer.usage_stats.values())
        
        if total == 0:
            print("Aucune donnée d'usage disponible")
            return
        
        # Lettres les plus utilisées
        sorted_letters = sorted(
            self.transformer.usage_stats.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        print("\nLettres les plus fréquentes:")
        for letter, count in sorted_letters:
            print(f"{letter}: {count} fois ({count/total:.1%})")
        
        # Distribution des couleurs
        bg_dist = self.user_profile['color_prefs']['background']
        text_dist = self.user_profile['color_prefs']['text']
        print(f"\nPréférence de couleur: Fond {bg_dist+1}, Texte {text_dist+1}")

# Point d'entrée principal
if _name_ == "_main_":
    logger.info("Démarrage du système d'adaptation pour dyslexie")
    try:
        assistant = DyslexiaAssistant()
        assistant.run()
        logger.info("Système arrêté normalement")
    except Exception as e:
        logger.exception("ERREUR CRITIQUE")
        print(f"Une erreur grave est survenue: {str(e)}")
        print("Veuillez consulter le fichier de log pour plus de détails")