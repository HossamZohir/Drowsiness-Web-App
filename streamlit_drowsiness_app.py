# streamlit_drowsiness_app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import sys

# === CONFIG ===
MODEL_PATH = "best_model.pth"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_NAME = "mobilenetv3_large_100"
CLASS_NAMES = ["Drowsy", "Non Drowsy"]
CONFIDENCE_THRESHOLD = 0.80

# === Resource Helper (for local + cloud deployment) ===
def get_resource_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(__file__), filename)

# === Load Model (cache to avoid reloading) ===
@st.cache_resource
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=False)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.out_features == 1000:
            in_features = module.in_features
            setattr(model, name, torch.nn.Linear(in_features, len(CLASS_NAMES)))
            break
    model.load_state_dict(torch.load(get_resource_path(MODEL_PATH), map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Video Transformer with Smoothing ===
class DrowsinessDetector(VideoTransformerBase):
    def __init__(self):
        cascade_file = get_resource_path(CASCADE_PATH)
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

        self.frame_count = 0
        self.prediction_interval = 10  # Predict every 10 frames
        self.prediction_history = []
        self.max_history = 10
        self.last_label = "Waiting..."

    def transform(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = img[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            if self.frame_count % self.prediction_interval == 0:
                input_tensor = transform(pil_img).unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, pred = torch.max(probs, 1)
                    label = CLASS_NAMES[pred.item()] if conf.item() >= CONFIDENCE_THRESHOLD else "Uncertain"

                self.prediction_history.append(label)
                if len(self.prediction_history) > self.max_history:
                    self.prediction_history.pop(0)

                # Majority vote
                self.last_label = max(set(self.prediction_history), key=self.prediction_history.count)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, self.last_label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return img

# === Streamlit UI ===
st.set_page_config(page_title="Drowsiness Detection", layout="centered")
st.title("😴 Real-Time Driver Drowsiness Detection")
st.markdown("Using MobileNetV3 + Face Detection with Smoothed Output (mobile friendly)")

webrtc_streamer(
    key="drowsiness",
    video_transformer_factory=DrowsinessDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
