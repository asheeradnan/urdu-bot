"""
Streamlit Urdu Chatbot App
Author: Asheer Adnan
"""

import streamlit as st
import torch
import os
import json
import pickle
from pathlib import Path
import sys
import requests

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Urdu Chatbot 🤖",
    page_icon="💬",
    layout="wide"
)

# ============================================
# PREPROCESSOR CLASS (needed for pickle)
# ============================================
class UrduPreprocessor:
    """Class used in preprocessor.pkl"""
    def __init__(self, vocab=None):
        self.vocab = vocab or {}

    def encode(self, text):
        # Dummy encoding — replace with real logic
        return torch.tensor([[1, 2, 3]])

    def decode(self, indices):
        # Dummy decoding — replace with real logic
        return "یہ ایک فرضی جواب ہے۔"

# ============================================
# DOWNLOAD MODEL WEIGHTS FROM GITHUB
# ============================================
def download_model_weights():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.pth")

    if not os.path.exists(model_path):
        url = "https://github.com/<username>/<repo>/releases/download/<tag>/best_model.pth?raw=true"
        with st.spinner("📥 Downloading model weights from GitHub..."):
            r = requests.get(url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("✅ Model weights downloaded!")
    else:
        st.info("✅ Model weights already exist, skipping download.")

download_model_weights()

# ============================================
# LOAD MODEL (CACHED)
# ============================================
@st.cache_resource
def load_model():
    try:
        model_path = Path("models/best_model.pth")
        config_path = Path("model_config.json")      # directly from repo
        preproc_path = Path("preprocessor.pkl")      # directly from repo

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        # Import Transformer model
        sys.path.append(str(Path(__file__).parent))
        from model import Transformer

        # Initialize model
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

        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Load preprocessor
        with open(preproc_path, "rb") as f:
            preprocessor = pickle.load(f)

        return model, preprocessor, device

    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, UrduPreprocessor(), None

# Initialize model
with st.spinner("🔄 Loading Urdu Chatbot model..."):
    model, preprocessor, device = load_model()

# ============================================
# RESPONSE GENERATION
# ============================================
def generate_response(text):
    """Generate chatbot response."""
    if not model:
        return "⚠️ Model not loaded. Please refresh the page."

    try:
        input_tensor = preprocessor.encode(text).unsqueeze(0).to(device)
        with torch.no_grad():
            # Replace with your actual model.generate() logic
            output_indices = model.generate(input_tensor)
        response = preprocessor.decode(output_indices[0])
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

user_input = st.text_input("اپنا پیغام یہاں لکھیں (Type your message in Urdu):")

if st.button("Send"):
    if user_input.strip():
        bot_reply = generate_response(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", bot_reply))
    else:
        st.warning("⚠️ براہ کرم کوئی پیغام درج کریں (Please type a message).")

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"🧑‍💬 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & PyTorch | Developed by **Asheer Adnan (2025)**")
