from re import S
import streamlit as st
import pandas as pd

model = load_model("") # cargamos el modelo

st.title("Emotion Reader")  #  ponemos titulo
st.markdown(***Powered by :[Karen Paiva Leon](https://github.com/infokaren20), [Nahuel vazquez](https://github.com/najuvgz) y [Adam Candalija Naranjo](https://github.com/AdamCN10)***)

st.divider()

val1 = st.text_area("write here your text") # cargamos las variables - texto
val2 = st.file_uploader('upload here your csv archive ', type='csv') # cargamos las variables - csv

emotions = {2 : 'joy',
            3 : 'sadness',
            0 : 'anger',
            1 : 'fear',
            4 : 'surprise'}

if st.button("Predict Emotion from text"): # predecimos la emocion del texto
    if val1 != "":
        pred = model.predict([val1])
        st.success(f'The emotion is: {emotions[pred[0]]}')
    else:
        st.error("Please write a text")
    
    