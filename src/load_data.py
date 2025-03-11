import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder  
import mlflow
class DataProcessor:
    def __init__(self):
        self.raw = pd.read_csv("data/fake_or_real_news.csv")
        self.features = np.array(self.raw["title"])
        self.labels = np.array(self.raw["label"].astype('category'))
        self.vectorizer = CountVectorizer()
        self.X = self.vectorizer.fit_transform(self.features)
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.labels)
        self.dataset = mlflow.data.from_pandas(self.raw, source="data/fake_or_real_news.csv", name="Fake News", targets="label")

    def transform_features(self, features):
        return self.vectorizer.transform(features)


    def transform_labels(self, labels):
        return self.label_encoder.transform(labels)
    
    def inverse_transform_labels(self, labels):
        return self.label_encoder.inverse_transform(labels)
    
