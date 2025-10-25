"""
Streamlit Urdu Chatbot App
Author: Asheer Adnan
"""

import streamlit as st
import torch
import os
import pickle
from pathlib import Path

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Urdu Chatbot 🤖",
    page_icon="💬",
    layout="wide"
)

# ============================================
# PREPROCESSOR CLASS
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
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    model_path = Path("models/best_model.pth")
    preproc_path = Path("models/preprocessor.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full model
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
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

# ============================================
# INITIALIZE MODEL
# ============================================
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
        # Dummy placeholder for actual model inference
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

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"🧑‍💬 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & PyTorch | Developed by **Asheer Adnan (2025)**")
