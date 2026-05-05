import pandas as pd

data = pd.read_csv("C:/Users/Sachin/Desktop/Enginow_project2_sentiment-analysis-project/Data/reviews.csv")

print("Shape:", data.shape)
print("\nColumns:", data.columns)

print("\nSample Data:\n", data.head())

print("\nValue Count:\n", data['sentiment'].value_counts())