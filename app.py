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
# MODEL DOWNLOAD FUNCTION
# ============================================
def download_model_from_drive():
    """Download and extract model files from Google Drive (if not already present)."""
    model_dir = "models"
    zip_path = "model_folder.zip"

    # ✅ Your Google Drive file ID (from shared link)
    file_id = "1wPump7hM0JDtUk7Gvz0Q_AwLIIYf2Kgv"
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_dir):
        with st.spinner("📦 Downloading model files from Google Drive... (first time only)"):
            gdown.download(url, zip_path, quiet=False)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(model_dir)
            os.remove(zip_path)
            st.success("✅ Model files downloaded and extracted successfully!")
    else:
        print("✅ Model folder already exists — skipping download.")


# Run the download function
download_model_from_drive()


# ============================================
# LOAD MODEL (CACHED)
# ============================================
@st.cache_resource
def load_model():
    """Load model, preprocessor, and config (cached for performance)."""
    try:
        # File paths
        model_path = Path("models/best_model.pth")
        config_path = Path("models/model_config.json")
        preproc_path = Path("models/preprocessor.pkl")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        # Import your Transformer class
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
        return None, None, None


# Load model
with st.spinner("🔄 Loading Urdu Chatbot model..."):
    model, preprocessor, device = load_model()


# ============================================
# RESPONSE GENERATION
# ============================================
def generate_response(text):
    """Generate chatbot response."""
    if not model:
        return "⚠️ Model not loaded. Please refresh the page."

    # Dummy response placeholder
    # 👉 Replace this with your inference function if available
    return f"🤖 {text} کے بارے میں بات کرتے ہیں۔"


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

# ============================================
# DISPLAY CHAT
# ============================================
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"🧑‍💬 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")

st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit & PyTorch | Developed by **Asheer Adnan (2025)**"
)
