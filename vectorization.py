import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("C:/Users/Sachin/Desktop/Enginow_project2_sentiment-analysis-project/Data/reviews.csv")

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

data['clean_text'] = data['review'].apply(clean_text)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data['clean_text'])

print("Feature Names:\n", vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:\n", X.toarray())