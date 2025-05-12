import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
import os
import csv
from datetime import datetime
from torchvision import transforms, models

# --- Konfiguracja aplikacji ---
PAGE_TITLE = "🎨 Rysowanie i Rozpoznawanie AI"
CANVAS_SIZE = 224
FEEDBACK_CSV = os.path.join("feedback", "feedback.csv")
MODEL_PATH = "f1_track_layout_resnet18_v1_unfiltered_best.pth"
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

@st.cache_resource
def load_model(path: str) -> torch.nn.Module:
    """Ładuje wytrenowany model ResNet18."""
    base = models.resnet18(weights=None)
    in_features = base.fc.in_features
    base.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(256, len(CLASS_NAMES))
    )
    state = torch.load(path, map_location='cpu')
    weights = state.get('model_state_dict', state) if isinstance(state, dict) else state
    base.load_state_dict(weights)
    base.eval()
    return base

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
    # Save image
    track_name = model_pred if correct else user_label
    img_path = os.path.join("data", "filtered_images", track_name, f"{timestamp}.png")
    img.save(img_path)
    # Append CSV
    with open(FEEDBACK_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, img_path, model_pred, correct, user_label])

def main():
    # Ustawienia strony
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    init_feedback_storage()

    # Wczytanie zasobów
    model = load_model(MODEL_PATH)
    preprocessor = get_preprocessor(CANVAS_SIZE)

    # Układ: dwie kolumny (płótno | wyniki)
    col_canvas, col_results = st.columns([1, 1])

    with col_canvas:
        st.subheader("✍️ Narysuj tutaj")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=4,
            stroke_color="#000000",
            background_color="#FFFFFF",
            drawing_mode="freedraw",
            update_streamlit=True,
            display_toolbar=False,
            height=600,
            width=600,
            key="canvas"
        )
        # Przycisk do uruchomienia predykcji
        run_predict = st.button("🚀 Rozpoznaj rysunek")

    with col_results:
        if run_predict and canvas_result.image_data is not None:
            # Przygotowanie obrazu
            img_array = canvas_result.image_data[:, :, :3].astype('uint8')
            img = Image.fromarray(img_array)

            # Predykcja
            with st.spinner("Analiza obrazu..."):
                probabilities = predict(img, model, preprocessor)

            top5 = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5])

            # Najlepsza klasa
            best_class = max(probabilities, key=probabilities.get)

            # Wyniki
            st.subheader("🤖 Wynik rozpoznawania")
            st.write(f"**Klasa:** {best_class}")

            st.subheader("✅ Czy model się nie myli?")
            correct = st.radio("Model poprawnie rozpoznał tor?", options=[True, False], index=0)
            user_label = best_class
            if not correct:
                user_label = st.selectbox("Wybierz poprawny tor:", options=CLASS_NAMES)

            if st.button("💾 Zapisz feedback"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                append_feedback(timestamp, img, best_class, correct, user_label)
                st.success("Dzięki za feedback!🔥")

            # Tabela z 5 najlepszymi
            st.subheader("🏆 Top 5 wyników")
            st.table(top5)

            # Wykres słupkowy ze wszystkimi klasami
            st.subheader("📊 Pełna rozkład prawdopodobieństwa")
            st.bar_chart(probabilities)
        else:
            st.info("Narysuj coś i kliknij 'Rozpoznaj rysunek'.")

if __name__ == '__main__':
    main()
