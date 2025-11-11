# Project summary:
This project reads a prompt from the user, classifies the prompt sentiment (positive/negative/neutral), then generates a paragraph that matches that sentiment. It uses Hugging Face Transformers: DistilBERT finetuned on SST-2 for sentiment and GPT-2 Medium for generation. UI is built using Streamlit.

# Components:
*sentiment.py* — sentiment analyzer using distilbert-base-uncased-finetuned-sst-2-english.

*generator.py* — text generator using GPT-2 Medium.

*app.py* — Streamlit frontend.

# Datasets & Models:
*Sentiment model*: distilbert-base-uncased-finetuned-sst-2-english — pretrained and finetuned on SST-2 (Stanford Sentiment Treebank), which is a standard dataset for sentiment classification (binary pos/neg). Treated low-confidence(score < 0.6) predictions in both classes as neutral.

*Generation model*: GPT-2 Medium. In this example; we use prompt-conditioning:

e.g., Write a short, upbeat and supportive paragraph about: {user_prompt}

# Challenges and Problems Faced:

- *Mismatch between prompt and generation*: GPT-2 medium sometimes ignores sentiment and does not produce the exact results as it is pre-trained on WebText, fine-tuning is required to tailor the model specifically to the user's prompts.

- *Model size & latency*: Larger models give better results but require more RAM/compute. 

- *Neutral detection*: SST-2 is binary; neutral is synthetic here (low-confidence). For robust neutral detection, a model trained on 3-way sentiment labels (e.g., SST-5) or larger datasets should be preferred.

# Setup Instructions:

## 1. Clone or Download the Repository:
run the commands:

- git clone https://github.com/Abeeha5/sentiment-text-generator
- cd sentiment-text-generator

## 2. Create and Activate a Virtual Environment:
then run the following commands:

- python -m venv venv
- source .venv/bin/activate    ( For Mac/Linux)
- .venv\Scripts\activate       (For Windows)

## 3. Install Required Libraries:
Create a requirements.txt with:

- streamlit>=1.20

- transformers>=4.30.0

- torch>=2.0.0

- sentencepiece

- numpy

Then run: *pip install -r requirements.txt*

## Run App (Frontend):
run the command: *streamlit run app.py*

This will open a browser tab like:
http://localhost:8501


You’ll see:

- A text area to enter a prompt

- Automatic sentiment detection

- Generated paragraph aligned with sentiment

