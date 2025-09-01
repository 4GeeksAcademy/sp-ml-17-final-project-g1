import regex as re
import pickle
import streamlit as st
import pandas as pd
from nltk import download
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gdown
file_id = '1uOJUzlUJTELtjOTmRCTZhaZpYB6c-hIW'
model_file = "temp_model.pkl"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, model_file, quiet=False)
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
lemmatizer = WordNetLemmatizer()S
download("stopwords")
stop_words = stopwords.words("english")
def lemmatize_text(words, lemmatizer=lemmatizer):
    tokens = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 2]
    return tokens
#st.image("ruta/a/tu/logo.png", width=120)
#Mejorar el diseño con columnas - ó mejor poner pestañas
#3. Agregar colores y emojis
st.title(":cerebro: Emotion Reader")
# 7. Personalizar el tema
# Puedes modificar el archivo .streamlit/config.toml
# para cambiar colores y fuentes del tema de Streamlit.
#  Agregar animaciones o efectos
# Puedes usar st.balloons() o st.snow() para celebrar cuando se hace una predicción exitosa.
st.balloons()
emotions = {"2" : 'joy :sonrojo:',
            "3" : 'sadness :lloros:',
            "0" : 'anger :enfado:',
            "1" : 'fear :grito:',
            "4" : 'surprise :asombrado:'}
st.title("Emotion Reader")  #  ponemos titulo
st.markdown('Powered by :[Karen Paiva Leon](https://github.com/infokaren20), [Nahuel vazquez](https://github.com/najuvgz) y [Adam Candalija Naranjo](https://github.com/AdamCN10)')
st.divider()
val1 = st.text_area("write here your text. you can try with: I am sad ") # cargamos las variables - texto
if st.button("Predict Emotion from text"): # predecimos la emocion del texto
    if val1 != "":
        val1 = {'text' : [val1]}
        val1 = pd.DataFrame.from_dict(val1)
        val1["text"] = val1["text"].apply(preprocess_text)
        val1["text"] = val1["text"].apply(lemmatize_text)
        val1 = val1['text']
        val1 = [" ".join(tokens) for tokens in val1]
        val1 = vectorizer.transform(val1).toarray()
        pred = model.predict([val1])
        st.success(f'The emotion is: {emotions[pred[0]]}')
    else:
        st.error("Please write a text")
'''
val2 = st.file_uploader('upload here your csv archive ', type='csv') # cargamos las variables - csv
if st.button("Predict Emotion from csv"):
    if val2 is not None:
        data = pd.read_csv(val2, names=['text'])
        for i in range(len(data)):
         data['emotion'][i] = model.predict(data['text'][i])
         data['emotion'] = data['emotion'].map(emotions)
         st.success("Emotions predicted successfully")
         st.dataframe(data)
         csv = data.to_csv(index=False).encode('utf-8')
         st.download_button(label="Download data as CSV", data=csv, file_name='predicted_emotions.csv', mime='text/csv')
    else:
            st.error("The uploaded CSV must contain a 'text' column")
if st.DownloadButton("Download ", data=""({'text': ['I am very happy today', 'I am very sad today', 'I am very angry today', 'I am very scared today', 'I am very surprised today']}), file_name='sample.csv', mime='text/csv'):
    st.write("Sample file downloaded")
'''
'''