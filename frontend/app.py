import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sentira AI | Dev Dharmesh Patel",
    page_icon="🧠",
    layout="wide"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border: 1px solid #0056b3;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL & TOKENIZER ---
@st.cache_resource
def load_model():
    model_path = "emotion_model" 
    # use_fast=False ensures compatibility with local environments
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

try:
    tokenizer, model = load_model()
    classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- SIDEBAR (RESUME & CONTACT) ---
with st.sidebar:
    st.markdown("# 🧠 Sentira AI")
    st.markdown("### *Emotion Recognition Intelligence System*")
    st.markdown("---")
    st.markdown("### 👤 Developer")
    st.info("**Dev Dharmesh Patel**")
    st.write("📧 [devdpatel0005@gmail.com](mailto:devdpatel0005@gmail.com)")
    st.write("🔗 [GitHub Profile](https://github.com/devpatel0005)")
    
    st.markdown("---")
    st.markdown("### 🛠️ Core Technology")
    st.write("- **Architecture:** RoBERTa-Base")
    st.write("- **Backend:** PyTorch / Transformers")
    st.write("- **Domain:** Natural Language Processing")
    
    st.markdown("---")
    st.caption("v1.0.0 | Focused on High-Accuracy Sentiment Analysis")

# --- MAIN UI ---
st.title("Sentira AI: Deep Emotion Intelligence")
st.markdown("##### Transform raw text into actionable emotional insights using state-of-the-art Transformers.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Analysis Terminal")
    
    # Quick Test Dropdown
    st.write("Pick a test case or type your own:")
    example_input = st.selectbox("Pre-loaded examples:", [
        "Select an example...",
        "I just got my dream job offer and I couldn't be happier!",
        "Oh great, another flat tire. Just what I needed today.",
        "I heard a strange noise downstairs and my heart started racing.",
        "I am absolutely mesmerized by the starry sky tonight.",
        "I can't believe you actually did that, I'm so shocked!"
    ])

    default_text = "" if example_input == "Select an example..." else example_input

    user_input = st.text_area(
        "Input Text:",
        value=default_text,
        placeholder="Type a sentence here...",
        height=150
    )
    
    predict_button = st.button("Run Intelligence Analysis")

with col2:
    st.subheader("Emotional Profile")
    if predict_button and user_input.strip() != "":
        # Preprocessing
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calculations
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).flatten()
        prediction = torch.argmax(logits, dim=1).item()
        conf_score = probs[prediction].item()
        
        # Big Metric Display
        st.metric(label="Detected Emotion", value=classes[prediction].upper(), delta=f"{conf_score:.2%} Confidence")
        
        # Visualization
        df = pd.DataFrame({
            'Emotion': [c.capitalize() for c in classes],
            'Confidence': probs.tolist()
        })
        
        fig = px.bar(df, x='Confidence', y='Emotion', orientation='h',
                     color='Confidence',
                     color_continuous_scale='Blues',
                     text_auto='.2%')
        
        fig.update_layout(
            showlegend=False, 
            height=350, 
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Confidence Probability",
            yaxis_title=None
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif predict_button:
        st.warning("Analysis failed: Please provide input text.")
    else:
        st.info("Awaiting input... The results and confidence distribution will appear here.")

# --- FOOTER ---
st.markdown("---")
footer_col1, footer_col2 = st.columns([2, 1])
with footer_col1:
    st.markdown("© 2026 Dev Dharmesh Patel | Built for the 'Advanced Frontend Development and Deployment' Portfolio.")