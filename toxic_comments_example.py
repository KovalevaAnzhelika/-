#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
nltk.download('punkt')
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.metrics import plot_precision_recall_curve
import numpy as np
from sklearn.model_selection import GridSearchCV


# In[ ]:


df = pd.read_csv("./data/labeled.csv", sep=",")


# In[ ]:


df.shape


# In[ ]:


df.head(5)


# In[ ]:


df["toxic"] = df["toxic"].apply(int)


# In[ ]:


df.head(5)


# In[ ]:


df["toxic"].value_counts()


# In[ ]:


for c in df[df["toxic"] == 1]["comment"].head(5):
    print(c)


# In[ ]:


for c in df[df["toxic"] == 0]["comment"].head(5):
    print(c)


# In[ ]:


train_df, test_df = train_test_split(df, test_size=500)


# In[ ]:


test_df.shape


# In[ ]:


test_df["toxic"].value_counts()


# In[ ]:


train_df["toxic"].value_counts()


# In[ ]:


sentence_example = df.iloc[1]["comment"]
tokens = word_tokenize(sentence_example, language="russian")
tokens_without_punctuation = [i for i in tokens if i not in string.punctuation]
russian_stop_words = stopwords.words("russian")
tokens_without_stop_words_and_punctuation = [i for i in tokens_without_punctuation if i not in russian_stop_words]
snowball = SnowballStemmer(language="russian")
stemmed_tokens = [snowball.stem(i) for i in tokens_without_stop_words_and_punctuation]



# In[ ]:


print(f"Исходный текст: {sentence_example}")
print("-----------------")
print(f"Токены: {tokens}")
print("-----------------")
print(f"Токены без пунктуации: {tokens_without_punctuation}")
print("-----------------")
print(f"Токены без пунктуации и стоп слов: {tokens_without_stop_words_and_punctuation}")
print("-----------------")
print(f"Токены после стемминга: {stemmed_tokens}")
print("-----------------")


# In[ ]:


snowball = SnowballStemmer(language="russian")
russian_stop_words = stopwords.words("russian")

def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
    tokens = word_tokenize(sentence, language="russian")
    tokens = [i for i in tokens if i not in string.punctuation]
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]
    return tokens


# In[ ]:


tokenize_sentence(sentence_example)


# In[ ]:


vectorizer = TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))


# In[ ]:


features = vectorizer.fit_transform(train_df["comment"])


# In[ ]:


model = LogisticRegression(random_state=0)
model.fit(features, train_df["toxic"])


# In[ ]:


model.predict(features[0])


# In[ ]:


train_df["comment"].iloc[0]


# In[ ]:


model_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))),
    ("model", LogisticRegression(random_state=0))
]
)


# In[ ]:


model_pipeline.fit(train_df["comment"], train_df["toxic"])


# In[ ]:


model_pipeline.predict(["Привет, у меня все нормально"])


# In[ ]:


model_pipeline.predict(["Слушай не пойти ли тебе нафиг отсюда?"])


# In[ ]:


precision_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict(test_df["comment"]))


# In[ ]:


recall_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict(test_df["comment"]))


# In[ ]:


prec, rec, thresholds = precision_recall_curve(y_true=test_df["toxic"], probas_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1])


# In[ ]:


plot_precision_recall_curve(estimator=model_pipeline, X=test_df["comment"], y=test_df["toxic"])


# In[ ]:


np.where(prec > 0.95)


# In[ ]:


thresholds[374]


# In[ ]:


precision_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1] > thresholds[374])


# In[ ]:


recall_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1] > thresholds[374])


# In[ ]:


grid_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))),
    ("model", 
     GridSearchCV(
        LogisticRegression(random_state=0),
        param_grid={'C': [0.1, 1, 10.]},
        cv=3,
         verbose=4
        )
    )
])


# In[ ]:


grid_pipeline.fit(train_df["comment"], train_df["toxic"])


# In[ ]:


model_pipeline_c_10 = Pipeline([
    ("vectorizer", TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))),
    ("model", LogisticRegression(random_state=0, C=10.))
]
)


# In[ ]:


model_pipeline_c_10.fit(train_df["comment"], train_df["toxic"])


# In[ ]:


prec_c_10, rec_c_10, thresholds_c_10 = precision_recall_curve(y_true=test_df["toxic"], probas_pred=model_pipeline_c_10.predict_proba(test_df["comment"])[:, 1])


# In[ ]:


np.where(prec_c_10 > 0.95)


# In[ ]:


precision_score(y_true=test_df["toxic"], y_pred=model_pipeline_c_10.predict_proba(test_df["comment"])[:, 1] > thresholds_c_10[316])


# In[ ]:


recall_score(y_true=test_df["toxic"], y_pred=model_pipeline_c_10.predict_proba(test_df["comment"])[:, 1] > thresholds_c_10[316])

