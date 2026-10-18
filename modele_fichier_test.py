import pdfplumber #j'ai installé dans le Terminal pdfplumber pour traiter des pdf

# Chemin vers ton fichier PDF
chemin_pdf = r"d:\Project Rodrigue et Harini\texte essai python.pdf" #il faut que tu mettes l'emplcament de ton fichier texte essai python (le mien c'est celui qui est écrit en rouge( cf. libre office que je t envoyé))
chemin_txt = r"d:\Project Rodrigue et Harini\texte_essai_python_converti.txt" # c l'emplacement où la copie va être créer
chemin_sortie=r"d:\Project Rodrigue et Harini\texte_essai_python_modifié_dyslexia.txt"

# Extraction du texte
with pdfplumber.open(chemin_pdf) as pdf:
    texte_total = ""
    print(pdf.pages)#Nous donne le nb de pages pdf.pages est une liste
    print("**")#marquage
    for page in pdf.pages:
        print(page)#nous donne le num de la page actuek
        texte_total += page.extract_text() + "\n"

# Sauvegarde dans un fichier .txt
with open(chemin_txt, "w", encoding="utf-8") as fichier_texte:
    fichier_texte.write(texte_total)

print("Conversion terminée. Le texte a été enregistré dans le fichier .txt.")
print("**")


###

def conversion_texte_dico(chemin_txt):
    # Dictionnaire de correspondance
    dico = {'p': 'q', 'r': 's', 'a': 'o'}

    # Lecture du contenu du fichier
    with open(chemin_txt, 'r', encoding='utf-8') as fichier:
        texte = fichier.read()

    # Remplacement des caractères selon le dictionnaire
    texte_modifie = ''
    for lettre in texte:
        if lettre in dico:
            texte_modifie += dico[lettre] # sa valeur
        else:
            texte_modifie += lettre

    # Écriture du texte modifié dans le fichier de sortie
    with open(chemin_sortie, 'w', encoding='utf-8') as fichier_sortie:
        fichier_sortie.write(texte_modifie)

    print(f"Texte modifié enregistré dans : {chemin_sortie}")
###

print(conversion_texte_dico(chemin_txt))
    
    

    

    
    

