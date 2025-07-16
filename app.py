import streamlit as st
import joblib
import torch
import cv2
import numpy as np
from PIL import Image
from model import MaskFacialClassifier
from torchvision import transforms


try:
    classes = joblib.load('models/classes.pkl')
except FileNotFoundError:
    print("Erreur : fichier 'classes.pkl' introuvable.")
    exit()

@st.cache_resource
def load_model():
    model = MaskFacialClassifier()
    model.load_state_dict(torch.load("models/mask_face_detection.pth", map_location = torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5] * 3, std = [0.5] * 3)
])

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

st.title("Détection de masque facial en temps réel")

run = st.button('Lancer la détection')

FRAME_WINDOW = st.image([])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Impossible de lire la caméra.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        label, prob, color = predict_face(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{label} ({prob*100:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()