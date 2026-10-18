import cv2

# Dimensions du cadre (en pixels)
frame_width = 640
frame_height = 480

# Coordonnées du repère central (zone à viser avec le visage ou les yeux)
box_width = 200
box_height = 100
box_x1 = (frame_width - box_width) // 2
box_y1 = (frame_height - box_height) // 2
box_x2 = box_x1 + box_width
box_y2 = box_y1 + box_height

# Ouvre la caméra (0 = webcam par défaut)
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dessine un rectangle de repère (en vert)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

    # Affiche un petit texte explicatif
    cv2.putText(frame, 'Place your eyes/face in the box',
                (box_x1 - 50, box_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Affiche l'image dans une fenêtre
    cv2.imshow('Eye/Face Alignment Guide', frame)

    # Touche 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libère les ressources
cap.release()
cv2.destroyAllWindows()
