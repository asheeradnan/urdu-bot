"""
Streamlit Urdu Chatbot App
Author: Asheer Adnan
"""

import streamlit as st
import torch
import os
import sys
import json
import pickle
from pathlib import Path
import gdown
import zipfile
import shutil

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Urdu Chatbot 🤖",
    page_icon="💬",
    layout="wide"
)

# ============================================
# PREPROCESSOR CLASS (needed to load pickle)
# ============================================
class UrduPreprocessor:
    """Class used in preprocessor.pkl"""
    def __init__(self, vocab=None):
        self.vocab = vocab or {}

    def encode(self, text):
        # Convert text to dummy tensor (replace with real logic)
        return torch.tensor([[1, 2, 3]])

    def decode(self, indices):
        # Convert tensor or indices to string
        return "یہ ایک فرضی جواب ہے۔"

# ============================================
# DOWNLOAD MODEL FUNCTION
# ============================================
def download_model():
    """Download and extract model files from GitHub or Drive if missing."""
    model_dir = Path("models")
    zip_path = model_dir / "model_files.zip"
    file_id = "1wPump7hM0JDtUk7Gvz0Q_AwLIIYf2Kgv"  # replace with your Drive/GitHub link

    if not model_dir.exists() or not any(model_dir.iterdir()):
        st.info("📥 Downloading model files...")
        model_dir.mkdir(exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(zip_path), quiet=False)

        st.info("📦 Extracting model files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(model_dir)
        zip_path.unlink()

        # Flatten nested folders if present
        nested = model_dir / "models"
        if nested.exists() and nested.is_dir():
            for f in nested.iterdir():
                shutil.move(str(f), str(model_dir))
            shutil.rmtree(nested)

        st.success("✅ Model ready!")
    else:
        st.info("✅ Model folder already exists.")

# Ensure model files are present
download_model()
st.write("Files in models folder:", os.listdir("models"))

# ============================================
# LOAD MODEL FUNCTION
# ============================================
@st.cache_resource
def load_model():
    model_path = Path("models/best_model.pth")
    config_path = Path("models/model_config.json")
    preproc_path = Path("models/preprocessor.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        st.warning("⚠️ model_config.json not found. Using default config.")
        config = {
            "src_vocab_size": 5000,
            "tgt_vocab_size": 5000,
            "d_model": 512,
            "num_heads": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "d_ff": 2048,
            "max_seq_len": 128,
            "dropout": 0.1
        }

    # Load model safely
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            from model import Transformer
            model = Transformer(
                src_vocab_size=config["src_vocab_size"],
                tgt_vocab_size=config["tgt_vocab_size"],
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_encoder_layers=config["num_encoder_layers"],
                num_decoder_layers=config["num_decoder_layers"],
                d_ff=config["d_ff"],
                max_seq_len=config["max_seq_len"],
                dropout=config["dropout"]
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Full model
            model = checkpoint.to(device)

        model.eval()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        model = None

    # Load preprocessor
    try:
        with open(preproc_path, "rb") as f:
            preprocessor = pickle.load(f)
    except Exception:
        preprocessor = UrduPreprocessor()

    return model, preprocessor, device

# Load model
with st.spinner("🔄 Loading Urdu Chatbot model..."):
    model, preprocessor, device = load_model()

# ============================================
# GENERATE RESPONSE
# ============================================
def generate_response(text):
    if not model:
        return "⚠️ Model not loaded. Please refresh the page."

    try:
        input_tensor = preprocessor.encode(text).unsqueeze(0).to(device)
        # Replace this with actual generate logic
        response = preprocessor.decode([1,2,3])
    except Exception:
        response = "معاف کریں، میں ابھی جواب دینے سے قاصر ہوں۔"

    return response

# ============================================
# STREAMLIT UI
# ============================================
st.title("🤖 اردو چیٹ بوٹ | Urdu Conversational Chatbot")
st.markdown("Chat with an AI model trained on Urdu conversations!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("اپنا پیغام یہاں لکھیں (Type your message in Urdu):", key="input_field")

if st.button("Send", key="send_button"):
    if user_input.strip():
        bot_reply = generate_response(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", bot_reply))
    else:
        st.warning("⚠️ براہ کرم کوئی پیغام درج کریں (Please type a message).")

# Display chat
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"🧑‍💬 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & PyTorch | Developed by **Asheer Adnan (2025)**")

