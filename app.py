import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Emotion Detection | Dev Dharmesh Patel",
    page_icon="🎭",
    layout="wide"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL & TOKENIZER ---
@st.cache_resource
def load_model():
    # Path where you saved your model using model.save_pretrained()
    model_path = "emotion_model" 
    # use_fast=False is used to avoid common dependency errors on local machines
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

try:
    tokenizer, model = load_model()
    # Classes based on your emotion_detection.py script
    classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- SIDEBAR (RESUME READY SECTION) ---
with st.sidebar:
    st.title("👤 Developer Profile")
    st.markdown("### **Dev Dharmesh Patel**")
    st.write("📧 [devdpatel0005@gmail.com](mailto:devdpatel0005@gmail.com)")
    st.write("🔗 [GitHub Profile](https://github.com/devpatel0005)")
    
    st.markdown("---")
    st.markdown("### 🛠️ Technical Stack")
    st.info("""
    - **Model:** RoBERTa (Base)
    - **Framework:** PyTorch & HuggingFace
    - **Library:** Transformers
    - **Frontend:** Streamlit
    """)
    
    st.markdown("### 📊 Dataset")
    st.caption("Trained on the Twitter Emotions Dataset, specializing in short-form text sentiment analysis.")

# --- MAIN UI ---
st.title("🎭 Emotion Recognition Intelligence System")
st.write("Detecting nuances in human expression through Natural Language Processing.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Text Input")
    user_input = st.text_area(
        "Enter text to analyze:",
        placeholder="How are you feeling today?",
        height=200
    )
    
    predict_button = st.button("Analyze Emotion")

with col2:
    st.subheader("Prediction Results")
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
        
        # Display Primary Result
        st.metric(label="Primary Detected Emotion", value=classes[prediction].upper(), delta=f"{conf_score:.2%} Confidence")
        
        # Visualization
        df = pd.DataFrame({
            'Emotion': [c.capitalize() for c in classes],
            'Confidence': probs.tolist()
        })
        
        fig = px.bar(df, x='Confidence', y='Emotion', orientation='h',
                     title="Confidence Distribution",
                     color='Confidence',
                     color_continuous_scale='RdYlGn',
                     text_auto='.2%')
        
        fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    elif predict_button:
        st.warning("Please enter text to see the analysis.")
    else:
        st.info("Results and confidence charts will appear here after analysis.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: grey;">
    Built by Dev Dharmesh Patel | 2026 Emotion AI Project
    </div>
    """, 
    unsafe_allow_html=True
)