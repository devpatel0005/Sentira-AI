
# 🧠 Sentira AI: Deep Emotion Intelligence System
**Developed by:** [Dev Dharmesh Patel](https://github.com/devpatel0005)  
**Email:** [devdpatel0005@gmail.com](mailto:devdpatel0005@gmail.com)  
**Live Demo:** [Streamlit Cloud App]

---

## 📌 Project Overview
**Sentira AI** is a state-of-the-art Natural Language Processing (NLP) system designed to detect and categorize human emotions in social media discourse. Built using a fine-tuned **RoBERTa-base transformer**, this system goes beyond simple sentiment analysis (positive/negative) to identify six distinct emotional states: **Joy, Sadness, Anger, Fear, Love, and Surprise**.

The project leverages the **Hugging Face Transformers** library and is deployed via an interactive **Streamlit** dashboard, providing real-time emotional profiling of any input text.

## 🚀 Key Features
* **Deep Transformer Architecture:** Utilizes RoBERTa (Robustly Optimized BERT Approach), which is pre-trained on larger datasets with longer training times for superior language understanding.
* **High-Accuracy Classification:** Achieving **94%+ accuracy** on the Twitter Emotions Dataset, outperforming standard models like DistilBERT.
* **Real-time Inference:** A lightweight frontend that processes text inputs and provides immediate emotional insights.
* **Confidence Scoring:** Visualizes the model's certainty for each category using Plotly-based analytics.
* **Professional Deployment:** Engineered with MLOps best practices, including model caching and environment-isolated dependencies.

## 🛠️ Technical Stack
* **Model:** RoBERTa (Base)
* **Language:** Python 3.10+
* **Libraries:** PyTorch, Hugging Face Transformers, Pandas, NumPy
* **Visualization:** Plotly, Seaborn (for training metrics)
* **Frontend:** Streamlit

## 📊 Model Performance
During development, the model was rigorously evaluated using precision, recall, and F1-score metrics. While simpler architectures like DistilBERT struggled with nuanced classes (specifically "Surprise"), the fine-tuned RoBERTa model showed significant improvement in generalization across all emotional categories.

### Classification Report (Test Data)
| Emotion | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Sadness** | 0.97 | 0.98 | 0.97 |
| **Joy** | 0.95 | 0.96 | 0.95 |
| **Love** | 0.92 | 0.90 | 0.91 |
| **Anger** | 0.94 | 0.94 | 0.94 |
| **Fear** | 0.91 | 0.91 | 0.91 |
| **Surprise** | 0.89 | 0.85 | 0.87 |

## ⚙️ Installation & Usage
To run the Sentira AI system locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/devpatel0005/sentira-ai.git
   cd sentira-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📂 Repository Structure
* `app.py`: The Streamlit frontend and inference logic.
* `Emotion_detection.ipynb`: The core research notebook detailing data preprocessing, model training, and evaluation.
* `emotion_model/`: Directory containing the saved model weights and tokenizer configurations.
* `requirements.txt`: List of necessary Python packages.

---
*Developed as part of my Advanced NLP Portfolio.*
