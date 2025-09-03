import regex as re
import random
import pickle
import streamlit as st
import pandas as pd
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
#import gdown
#import dropbox
import requests

def dropbox_download(share_url, output_path, quiet=False):
    """
    Función similar a gdown.download pero para Dropbox
    """
    
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        if not quiet:
            print(f"Descarga completada: {output_path}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error en la descarga: {e}")
        raise


file_id = {0 : '1uOJUzlUJTELtjOTmRCTZhaZpYB6c-hIW',
           1 : '1zLDmDIpVlnThs35RqerwvagUYAP9y7Vb',
           2 : '1xujpjVFWNylQseFOrLyxFNM2frxhmUXg',
           3 : '1mqvXZD6YSRMNBi2qPrttczeaLIaKpIgR',
           4 : '1OXBoEZxakl9N_E6TakKQTpjdQefiJvmM'
}
model_file = "temp_model.pkl"
#url = f"https://drive.google.com/uc?id={file_id[random.randint(1,4)]}"
url = 'https://www.dropbox.com/scl/fi/h8s8vixvk314vnl316syp/random-forest.pkl?rlkey=vcjvl5mvfplca9r7migv5hc3u&st=uaxgqnnq&dl=1'
#gdown.download(url, model_file, quiet=False, use_cookies=False)
dropbox_download(url, model_file, quiet=False)
with open(model_file, 'rb') as file:
            model = pickle.load(file)

        
with open('C:/Users/adamc/Documents/GitHub/sp-ml-17-final-project-g1/models/text-vectorizer.pkl', 'rb') as file:
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

#st.image("ruta/a/tu/logo.png", width=120)

#Mejorar el diseño con columnas - ó mejor poner pestañas

#3. Agregar colores y emojis
st.title(":brain: Emotion Reader")



# 7. Personalizar el tema
# Puedes modificar el archivo .streamlit/config.toml
# para cambiar colores y fuentes del tema de Streamlit.
#  Agregar animaciones o efectos
# Puedes usar st.balloons() o st.snow() para celebrar cuando se hace una predicción exitosa.
st.balloons()



emotions = {2 : 'joy',
            3 : 'sadness',
            0 : 'anger',
            1 : 'fear',
            4 : 'surprise'}


st.title("Emotion Reader")  #  ponemos titulo
st.markdown('Powered by :[Karen Paiva Leon](https://github.com/infokaren20), [Nahuel vazquez](https://github.com/najuvgz) y [Adam Candalija Naranjo](https://github.com/AdamCN10)')
st.divider()


val1 = st.text_area("write here your text. you can try with: I am sad ") # cargamos las variables - texto
if st.button("Predict Emotion from text"): # predecimos la emocion del texto
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
    else:
        st.error("Please write a text")


val2 = st.file_uploader('upload here your csv archive ', type='csv') # cargamos las variables - csv
csv = None
if st.button("Predict Emotion from csv"):
    if val2 is not None:
        data = pd.read_csv(val2, names=['text']).drop(0).reset_index(drop=True)
        data["text"] = data["text"].apply(preprocess_text)
        data["text"] = data["text"].apply(lemmatize_text)
        word_list = data["text"]
        word_list = [" ".join(tokens) for tokens in word_list]
        X = vectorizer.transform(word_list).toarray()
        emotions_predict = model.predict(X)
        data['emotion'] = pd.Series(emotions_predict).map(emotions)
        st.success("Emotions predicted successfully")
        st.dataframe(data)
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download data as CSV", data=csv, file_name='predicted_emotions.csv', mime='text/csv')
    else:
        st.error("The uploaded CSV must contain only a 'text' column")
#if st.download_button("Download predicted .csv", data=csv, file_name='predicted_emotions.csv'):
#    st.write("Sample file downloaded")
