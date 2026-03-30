import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Configuration
st.set_page_config(page_title="Emotion Detection", page_icon="🎭")

# 2. Load the Model and Tokenizer
@st.cache_resource # Cache the model so it doesn't reload on every interaction
def load_model():
    model_path = "emotion_model" # Ensure this matches your saved path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Define classes as per your original training script
classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# 3. Streamlit UI Elements
st.title("🎭 Emotion Recognition App")
st.write("Enter text below to detect the underlying emotion using a fine-tuned RoBERTa model.")

user_input = st.text_area("Input Text:", placeholder="Type something here...")

if st.button("Predict Emotion"):
    if user_input.strip() != "":
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process Results
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        emotion = classes[prediction]
        
        # Display Results
        st.success(f"Predicted Emotion: **{emotion.upper()}**")
        
        # Optional: Show confidence scores
        probs = torch.nn.functional.softmax(logits, dim=1).flatten()
        st.write("### Confidence Scores:")
        for i, prob in enumerate(probs):
            st.write(f"{classes[i].capitalize()}: {prob:.2%}")
            st.progress(float(prob))
    else:
        st.warning("Please enter some text first.")