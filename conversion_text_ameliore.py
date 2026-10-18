import pdfplumber #j'ai installé dans le Terminal pdfplumber pour traiter des pdf

# Chemin vers ton fichier PDF
chemin_entree_pdf = r"d:\\Project Rodrigue et Harini\\texte essai python.pdf" #il faut que tu mettes l'emplacement de ton fichier texte essai python
chemin_entree_txt = r"d:\\Project Rodrigue et Harini\\texte_essai_python_converti.txt" # emplacement de la copie en texte brut
chemin_sortie_txt = r"d:\\Project Rodrigue et Harini\\texte_essai_python_modifié_dyslexia.txt" # texte modifié pour dyslexie

# Extraction du texte
with pdfplumber.open(chemin_entree_pdf) as pdf:
    texte_total = ""
    print(pdf.pages)  # Affiche la liste des pages
    print("")       # Marqueur
    for page in pdf.pages:
        print(page)   # Affiche chaque page
        texte_total += page.extract_text() + "\n"

# Sauvegarde brute dans un fichier .txt
with open(chemin_entree_txt, "w", encoding="utf-8") as fichier_texte:
    fichier_texte.write(texte_total)

print("Copie brute enregistrée dans :", chemin_entree_txt)

# --- Morphisme typographique pour dyslexie ---

import tkinter as tk
from tkinter import simpledialog

# Substitutions Unicode par intensité (1 à 5)
map_intensity = {
    1: {'b': 'ʙ', 'd': 'ð'},
    2: {'b': 'ʙ', 'd': 'ð', 'p': 'þ', 'q': 'ɋ'},
    3: {'b': 'ʙ', 'd': 'ð', 'p': 'þ', 'q': 'ɋ', 'n': 'ɴ', 'u': 'ᴜ'},
    4: {'b': 'ʙ', 'd': 'ð', 'p': 'þ', 'q': 'ɋ', 'n': 'ɴ', 'u': 'ᴜ', 'm': 'ᴍ', 'w': 'ᴡ'},
    5: {'b': 'ʙ', 'd': 'ð', 'p': 'þ', 'q': 'ɋ', 'n': 'ɴ', 'u': 'ᴜ', 'm': 'ᴍ', 'w': 'ᴡ', 'i': 'ɪ', 'l': 'ʟ'}
}

# Interface graphique pour choisir l'intensité
root = tk.Tk()
root.withdraw()  # Masque la fenêtre principale
intensite = simpledialog.askinteger(
    "Niveau de transformation",
    "Choisissez un niveau d'intensité (1 = léger, 5 = fort)",
    minvalue=1, maxvalue=5
)

if intensite is None:
    print("Aucune intensité choisie. Annulation.")
    exit()

sub_map = map_intensity.get(intensite, {})

# Appliquer les substitutions au texte
texte_modifie = ""
for caractere in texte_total:
    if caractere.lower() in sub_map:
        nouveau = sub_map[caractere.lower()]
        texte_modifie += nouveau.upper() if caractere.isupper() else nouveau
    else:
        texte_modifie += caractere

# Sauvegarde du texte transformé
with open(chemin_sortie_txt, "w", encoding="utf-8") as fichier_modifie:
    fichier_modifie.write(texte_modifie)

#print("Texte modifié enregistré dans :", chemin_sortie_txt)

# Conversion du sortie txt en sortie pdf
from fpdf import FPDF

# Écriture du fichier texte
with open(chemin_sortie_txt, "w", encoding="utf-8") as fichier_modifie:
    fichier_modifie.write(texte_modifie)

# Conversion en PDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

# Enregistrer la police Unicode
pdfmetrics.registerFont(TTFont("DejaVu", "fonts/DejaVuSans.ttf"))

# Écriture du fichier texte
with open(chemin_sortie_txt, "w", encoding="utf-8") as fichier_modifie:
    fichier_modifie.write(texte_modifie)

# Conversion en PDF
chemin_sortie_pdf = chemin_sortie_txt.replace(".txt", ".pdf")

c = canvas.Canvas(chemin_sortie_pdf, pagesize=A4)
c.setFont("DejaVu", 12)  # utilisation de la police Unicode

width, height = A4
x = 40
y = height - 40
line_height = 14

for ligne in texte_modifie.split("\n"):
    if y < 40:
        c.showPage()
        c.setFont("DejaVu", 12)
        y = height - 40
    c.drawString(x, y, ligne)
    y -= line_height

c.save()

