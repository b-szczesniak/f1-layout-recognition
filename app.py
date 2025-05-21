import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv
from datetime import datetime
from torchvision import transforms, models

PAGE_TITLE = "Rozpoznawanie toru po rysunku!!1!🔥🔥"
CANVAS_SIZE = 224
FEEDBACK_CSV = os.path.join("feedback", "feedback.csv")
MODEL_PATH = "f1_track_layout_resnet18_v1.pth"
CLASS_NAMES = [
    'Albert Park Circuit', 'Autódromo Hermanos Rodríguez', 'Bahrain International Circuit',
    'Baku City Circuit', 'Circuit Gilles Villeneuve', 'Circuit Zandvoort',
    'Circuit de Barcelona-Catalunya', 'Circuit de Monaco', 'Circuit de Spa-Francorchamps',
    'Circuit of the Americas', 'Hungaroring',
    'Imola (Autodromo Enzo e Dino Ferrari)', 'Interlagos (Autódromo José Carlos Pace)',
    'Jeddah Corniche Circuit', 'Las Vegas Street Circuit', 'Lusail International Circuit',
    'Marina Bay Street Circuit', 'Miami International Autodrome', 'Monza (Autodromo Nazionale Monza)',
    'Red Bull Ring', 'Shanghai International Circuit', 'Silverstone Circuit',
    'Suzuka International Racing Course', 'Yas Marina Circuit'
]

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes: int):
        super(EfficientNetModel, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model(path: str) -> nn.Module:
    """Ładuje wytrenowany model EfficientNet."""
    model = EfficientNetModel(num_classes=len(CLASS_NAMES))
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_resource
def get_preprocessor(size: int) -> transforms.Compose:
    """Tworzy transformacje obrazu dla modelu."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict(image: Image.Image, model: torch.nn.Module, preprocessor: transforms.Compose) -> dict:
    """Generuje predykcje klas i prawdopodobieństwa."""
    tensor = preprocessor(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    return {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}

def init_feedback_storage():
    os.makedirs("feedback", exist_ok=True)
    if not os.path.exists(FEEDBACK_CSV):
        with open(FEEDBACK_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'image_path', 'model_pred', 'user_correct', 'user_label'])

def append_feedback(timestamp: str, img: Image.Image, model_pred: str, correct: bool, user_label: str):
    track_name = model_pred if correct else user_label
    img_path = os.path.join("data", "filtered_images", track_name, f"{timestamp}.png")
    img.save(img_path)
    with open(FEEDBACK_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, img_path, model_pred, correct, user_label])

def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    init_feedback_storage()

    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'probabilities' not in st.session_state:
        st.session_state.probabilities = None
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'img' not in st.session_state:
        st.session_state.img = None
    if 'best_class' not in st.session_state:
        st.session_state.best_class = None

    model = load_model(MODEL_PATH)
    preprocessor = get_preprocessor(CANVAS_SIZE)

    col_canvas, col_results = st.columns([1, 1])

    with col_canvas:
        st.subheader("✍️ Narysuj tutaj")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=6,
            stroke_color="#000000",
            background_color="#FFFFFF",
            drawing_mode="freedraw",
            update_streamlit=True,
            display_toolbar=False,
            height=600,
            width=600,
            key=f"canvas_{st.session_state.canvas_key}",
        )
        
        def make_prediction():
            if canvas_result.image_data is not None:
                img_array = canvas_result.image_data[:, :, :3].astype('uint8')
                img = Image.fromarray(img_array)
                
                # Predykcja
                with st.spinner("Analiza obrazu..."):
                    probabilities = predict(img, model, preprocessor)
                
                st.session_state.prediction_made = True
                st.session_state.probabilities = probabilities
                st.session_state.img = img
                st.session_state.best_class = max(probabilities, key=probabilities.get)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Rozpoznaj rysunek"):
                make_prediction()
        with col2:
            if st.button("🧹 Wyczyść płótno"):
                st.session_state.canvas_key += 1
                st.session_state.prediction_made = False
                st.rerun()

    with col_results:
        if st.session_state.prediction_made:
            probabilities = st.session_state.probabilities
            img = st.session_state.img
            best_class = st.session_state.best_class
            
            top5 = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5])
            
            st.subheader("🤖 Wynik rozpoznawania")
            st.write(f"**Klasa:** {best_class}")
            
            st.subheader("✅ Czy model się nie myli?")
            correct = st.radio(
                "Model poprawnie rozpoznał tor?", 
                options=["Tak", "Nie"], 
                index=0,
                key="correct_radio"
            )
            correct = (correct == "Tak")
            
            user_label = best_class
            if not correct:
                user_label = st.selectbox("Wybierz poprawny tor:", options=CLASS_NAMES, key="label_select")

            
            if st.button("💾 Zapisz feedback", key="save_feedback_btn"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                append_feedback(timestamp, img, best_class, correct, user_label)
                st.success("Dzięki za feedback!🔥")
                st.session_state.prediction_made = False

            st.subheader("🏆 Top 5 wyników")
            st.table(top5)
            
            st.subheader("📊 Pełna rozkład prawdopodobieństwa")
            st.bar_chart(probabilities)
        else:
            st.info("Narysuj coś i kliknij 'Rozpoznaj rysunek'.")


if __name__ == '__main__':
    main()
