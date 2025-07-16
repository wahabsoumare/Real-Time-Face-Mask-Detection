import torch
import cv2
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms
from model import MaskFacialClassifier

# Chargement des classes
try:
    classes = joblib.load('models/classes.pkl')
except FileNotFoundError:
    print("Erreur : fichier 'classes.pkl' introuvable.")
    exit()

# Initialisation du modèle
model = MaskFacialClassifier()
try:
    model.load_state_dict(torch.load('models/mask_face_detection.pth', map_location = torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    exit()

# Préparation de la transformation des images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5] * 3, std = [0.5] * 3)
])

def predict_from_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)

    input_tensor = transform(image_pil).unsqueeze(0)  # Shape: (1, 3, 128, 128)

    # Prédiction
    with torch.no_grad():
        output = model(input_tensor)
        prob = output.squeeze().item()
        pred = int(prob >= 0.5)
        # predicted_class = classes[pred]
        translation = {
            'with_mask': 'Mask',
            'without_mask': 'No mask'
        }
        predicted_class = translation[classes[pred]]

    return predicted_class, prob

def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    color = (0, 0, 0)
    thickness = 2
    margin = 0
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    # end_x = pos[0] + txt_size[0][0] + margin
    # end_y = pos[1] - txt_size[0][1] - margin

    # cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, thickness, cv2.LINE_AA)

# Chargement du détecteur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture vidéo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la trame.")
        break

    # Conversion en niveaux de gris pour la détection de visage
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))

    for (x, y, w, h) in faces:
        # Rectangle autour du visage
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Prédire si le visage porte un masque ou pas
        face_roi = frame[y:y + h, x:x + w]
        predicted_class, prob = predict_from_frame(face_roi)

        draw_label(frame, f'{predicted_class} ({prob * 100:.2f}%)', (x, y - 10), (255, 0, 0))

    cv2.imshow('Mask Detection - Real Time', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()