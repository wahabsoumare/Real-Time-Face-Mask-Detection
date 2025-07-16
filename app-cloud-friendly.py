import streamlit as st
import joblib
import torch
import cv2
import numpy as np
from PIL import Image
from model import MaskFacialClassifier
from torchvision import transforms

# Chargement des classes
try:
    classes = joblib.load('models/classes.pkl')
except FileNotFoundError:
    st.error("Erreur : fichier 'classes.pkl' introuvable.")
    st.stop()

# Chargement du modèle
@st.cache_resource
def load_model():
    model = MaskFacialClassifier()
    model.load_state_dict(torch.load("models/mask_face_detection.pth", map_location = torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Prétraitement
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5] * 3, std = [0.5] * 3)
])

# Fonction de prédiction
def predict_face(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = output.squeeze().item()
        pred = int(prob >= 0.5)
        translation = {
            'with_mask': 'Mask',
            'without_mask': 'No mask'
        }
        label = translation[classes[pred]]
        color = (0, 255, 0) if pred == 0 else (0, 0, 255)
        
    return label, prob, color

st.title("Détection de masque facial (via image) 😷")

uploaded_file = st.file_uploader("📤 Téléversez une image (jpg, png)", type = ["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))

    # Détection des visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor  = 1.1, minNeighbors = 5)

    for (x, y, w, h) in faces:
        face = image_np[y:y + h, x:x + w]
        label, prob, color = predict_face(face)
        cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image_np, f'{label} ({prob * 100:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    st.image(image_np, caption = "🖼 Résultat", use_container_width = True)