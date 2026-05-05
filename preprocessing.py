import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv("C:/Users/Sachin/Desktop/Enginow_project2_sentiment-analysis-project/Data/reviews.csv")

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    words = text.split()
    
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stopwords.words('english')
    ]
    
    return " ".join(words)

data['clean_text'] = data['review'].apply(clean_text)

print(data[['review', 'clean_text']])