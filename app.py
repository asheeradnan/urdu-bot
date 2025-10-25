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
# PREPROCESSOR CLASS (needed to load pickle)
# ============================================
class UrduPreprocessor:
    """Fallback class if preprocessor.pkl is missing"""
    def encode(self, text):
        return torch.tensor([[1, 2, 3]])  # placeholder
    def decode(self, indices):
        return "یہ ایک فرضی جواب ہے۔"

# ============================================
# MODEL DOWNLOAD FUNCTION (GitHub Release)
# ============================================
def download_model_from_github():
    """Download and extract model files from GitHub release."""
    model_dir = "models"
    zip_path = "model_files.zip"

    # GitHub release URL (replace with your release URL)
    github_url = "https://github.com/yourusername/yourrepo/releases/download/v1.0/best_model.zip"

    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        with st.spinner("📦 Downloading model from GitHub..."):
            gdown.download(github_url, zip_path, quiet=False)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(model_dir)
            os.remove(zip_path)
            st.success("✅ Model downloaded and extracted successfully!")
    else:
        st.info("✅ Model folder already exists — skipping download.")

download_model_from_github()
st.write("Files in models folder:", os.listdir("models"))

# ============================================
# LOAD MODEL FUNCTION
# ============================================
@st.cache_resource
def load_model():
    try:
        model_dir = Path("models")
        model_path = model_dir / "best_model.pth"
        config_path = model_dir / "model_config.json"
        preproc_path = model_dir / "preprocessor.pkl"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config or fallback
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

        # Import Transformer
        sys.path.append(str(Path(__file__).parent))
        from model import Transformer

        # Initialize empty model
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

        # Load weights (supports full model or state_dict)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint.to(device)
        model.eval()

        # Load preprocessor
        if preproc_path.exists():
            with open(preproc_path, "rb") as f:
                preprocessor = pickle.load(f)
        else:
            st.warning("⚠️ preprocessor.pkl not found. Using placeholder.")
            preprocessor = UrduPreprocessor()

        return model, preprocessor, device

    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
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
    """Generate chatbot response using Transformer and preprocessor."""
    if not model or not preprocessor:
        return "⚠️ Model not loaded. Please refresh the page."

    try:
        input_tensor = preprocessor.encode(text).to(device)
        input_tensor = input_tensor.unsqueeze(0)  # add batch dim

        # Use generate() if your Transformer has it
        if hasattr(model, "generate"):
            output_indices = model.generate(input_tensor)
        else:
            # fallback: forward pass + argmax
            with torch.no_grad():
                output_logits = model(input_tensor, input_tensor)
                output_indices = output_logits.argmax(dim=-1)

        response = preprocessor.decode(output_indices[0])
    except Exception as e:
        print("Error generating response:", e)
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

if st.button("Send") and user_input.strip():
    bot_reply = generate_response(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", bot_reply))
elif st.button("Send"):
    st.warning("⚠️ براہ کرم کوئی پیغام درج کریں (Please type a message).")

# Display chat
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"🧑‍💬 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & PyTorch | Developed by **Asheer Adnan (2025)**")
