# E-Commerce Review Intelligence System

Transformer-based sentiment analysis pipeline for e-commerce reviews using DistilBERT and PyTorch.

## Project Highlights
- Fine-tunes `distilbert-base-uncased` for binary sentiment classification (`negative`, `positive`).
- Handles label imbalance with class-weighted cross-entropy.
- Uses reproducible training (`seed=42`) and stratified train/test split.
- Reports accuracy, macro-F1, classification report, and confusion matrix.
- Exports model and tokenizer artifacts for deployment-style inference.

## Problem Statement
E-commerce businesses receive large volumes of customer reviews. Manually reading all reviews is not scalable. This project automates sentiment classification to help teams monitor customer satisfaction and identify issues quickly.

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers + Datasets
- Scikit-learn
- Pandas + NumPy

## Notebook
- Main workflow: `E_commerce_Review_Intelligence_System.ipynb`

## Dataset Setup
Place your dataset file at:

`data/7817_1.csv`

Required columns:
- `reviews.text`
- `reviews.rating`

## How to Run
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run all cells in `E_commerce_Review_Intelligence_System.ipynb`.

## Output Artifacts
After training, artifacts are saved in `sentiment_model/`:
- Fine-tuned model weights
- Tokenizer files
- `label_mapping.json`
- `eval_metrics.json`

## Resume-Ready Talking Points
- Built an end-to-end NLP pipeline for sentiment intelligence on noisy e-commerce text.
- Improved minority-class sensitivity via weighted loss in a custom Hugging Face Trainer.
- Designed reproducible experimentation and exportable inference artifacts.
- Evaluated model performance with class-wise metrics and error-distribution analysis.

## Future Improvements
- Hyperparameter tuning (learning rate, epochs, batch size).
- Compare DistilBERT with RoBERTa/DeBERTa variants.
- Add experiment tracking (MLflow or Weights & Biases).
- Deploy as REST API for real-time review scoring.
