# sentiment analysis
from transformers import pipeline

class sentiment_analyzer:

    def __init__(self): 
        self.pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    
    def assign_labels(self, text):
        self.text = text

        if not text or not text.strip():
            print('Text not found')
            return ('NEUTRAL', 1.0)

        out = self.pipeline(text[:512])[0]
        label = out['label']
        score = out['score']

        if score < 0.6:
            label = 'NEUTRAL'
        
        return (label, score)
        