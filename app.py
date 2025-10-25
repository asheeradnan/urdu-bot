"""
Streamlit Urdu Chatbot App
Author: Asheer Adnan
"""

import streamlit as st
import torch
import gdown
import zipfile
import os
import json
import pickle
from pathlib import Path
import sys

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Urdu Chatbot 🤖",
    page_icon="💬",
    layout="wide"
)

# ============================================
# DOWNLOAD MODEL FUNCTION (GitHub Release)
# ============================================
def download_model_from_github():
    """Download .pth file from GitHub Releases if not present."""
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_file = model_dir + "/best_model.pth"
    github_url = "https://github.com/<your-username>/<repo>/releases/download/<tag>/best_model.pth"

    if not os.path.exists(model_file):
        with st.spinner("📦 Downloading model from GitHub Releases..."):
            gdown.download(github_url, model_file, quiet=False)
        st.success("✅ Model downloaded from GitHub Releases.")
    else:
        st.info("✅ Model already exists, skipping download.")

download_model_from_github()

# ============================================
# LOAD MODEL FUNCTION (PyTorch 2.6+ Safe)
# ============================================
@st.cache_resource
def load_model():
    """Load model, preprocessor, and config safely (PyTorch 2.6+)."""
    try:
        model_dir = Path("models")
        model_path = model_dir / "best_model.pth"
        config_path = model_dir / "model_config.json"
        preproc_path = model_dir / "preprocessor.pkl"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----------------------
        # Load config
        # ----------------------
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

        # ----------------------
        # Import Transformer
        # ----------------------
        sys.path.append(str(Path(__file__).parent))
        from model import Transformer  # your Transformer class

        # ----------------------
        # Initialize model
        # ----------------------
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

        # ----------------------
        # Load weights (trusted source!)
        # ----------------------
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # fallback if the whole model was saved
            model = checkpoint

        model.eval()

        # ----------------------
        # Load preprocessor
        # ----------------------
        if preproc_path.exists():
            with open(preproc_path, "rb") as f:
                preprocessor = pickle.load(f)
        else:
            st.warning("⚠️ preprocessor.pkl not found. Using placeholder.")
            class UrduPreprocessor:
                def encode(self, text): return torch.tensor([[1,2,3]])
                def decode(self, idx): return "یہ ایک فرضی جواب ہے۔"
            preprocessor = UrduPreprocessor()

        return model, preprocessor, device

    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        class UrduPreprocessor:
            def encode(self, text): return torch.tensor([[1,2,3]])
            def decode(self, idx): return "یہ ایک فرضی جواب ہے۔"
        return None, UrduPreprocessor(), None

# ============================================
# INITIALIZE MODEL
# ============================================
with st.spinner("🔄 Loading Urdu Chatbot model..."):
    model, preprocessor, device = load_model()

# ============================================
# RESPONSE GENERATION
# ============================================
def generate_response(text):
    """Generate chatbot response."""
    if not model or not preprocessor:
        return "⚠️ Model not loaded. Please refresh the page."
    try:
        input_tensor = preprocessor.encode(text).unsqueeze(0).to(device)
        with torch.no_grad():
            output_indices = model.generate(input_tensor)  # Replace with your generate method
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
