import regex as re
import random
import pickle
import streamlit as st
import pandas as pd
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

model_file = "../models/random-forest.pkl"
with open(model_file, 'rb') as file:
            model = pickle.load(file)

        
with open('../models/text-vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def preprocess_text(text):
    text = text.lower()
    #text = re.sub("https?|www", " ", text)
    text = re.sub(r'[^a-z ]', " ", text)
    text = re.sub(r'\s+', " ", text)
    return text.split()

download("wordnet")
lemmatizer = WordNetLemmatizer()

download("stopwords")
stop_words = stopwords.words("english")

def lemmatize_text(words, lemmatizer=lemmatizer):
    tokens = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 2]
    return tokens

st.title(":brain: Emotion Reader")

st.balloons()


emotions = {2 : 'joy 😄',
            3 : 'sadness 😭',
            0 : 'anger 😡',
            1 : 'fear 😨',
            4 : 'surprise 😲'}

image_emotion = {2 : 'joy',
                 3 : 'sadness',
                 0 : 'anger',
                 1 : 'fear',
                 4 : 'surprise'}


st.markdown('Powered by :[Karen Paiva Leon](https://github.com/infokaren20), [Nahuel vazquez](https://github.com/najuvgz) y [Adam Candalija Naranjo](https://github.com/AdamCN10)')
st.divider()
st.text('This model predicts emotions from text and csv with only a text column.')
st.image('../docs/asets/emotion-wheel.png')
st.text('This model only can preditct the following emotions: Joy, Sadness, Fear, Anger and Surprise.')

val1 = st.text_area("write here your text. you can try with: I am sad ") 
if st.button("Predict Emotion from text"):
    if val1.strip():
        val1 = {'text' : [val1]}
        val1 = pd.DataFrame.from_dict(val1)
        val1["text"] = val1["text"].apply(preprocess_text)
        val1["text"] = val1["text"].apply(lemmatize_text)
        val1 = val1['text']
        val1 = [" ".join(tokens) for tokens in val1]
        val1 = vectorizer.transform(val1).toarray()
        pred = model.predict(val1)
        st.success(f'The emotion is: {emotions[pred[0]]}')
        st.image(f'../docs/asets/{image_emotion[pred[0]]}.webp')
    else:
        st.error("Please write a text")


val2 = st.file_uploader('upload here your csv archive ', type='csv') 
csv = None
if st.button("Predict Emotion from csv"):
    if val2 is not None:
        data = pd.read_csv(val2, names=['text'])
        foo = data["text"].apply(preprocess_text)
        foo = foo.apply(lemmatize_text)
        word_list = foo
        word_list = [" ".join(tokens) for tokens in word_list]
        X = vectorizer.transform(word_list).toarray()
        emotions_predict = model.predict(X)
        data['emotion'] = pd.Series(emotions_predict).map(emotions)
        st.success("Emotions predicted successfully")
        st.dataframe(data)
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download data as CSV", data=csv, file_name='predicted_emotions.csv', mime='text/csv')
    else:
        st.error("Please upload a CSV with only a 'text' column")