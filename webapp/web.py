from re import S
from token import NAME
import streamlit as st
import pandas as pd

model = load_model("") 

emotions = {"2" : 'joy',
            "3" : 'sadness',
            "0" : 'anger',
            "1" : 'fear',
            "4" : 'surprise'}

st.title("Emotion Reader")  
st.markdown(***Powered by :[Karen Paiva Leon](https://github.com/infokaren20), [Nahuel vazquez](https://github.com/najuvgz) y [Adam Candalija Naranjo](https://github.com/AdamCN10)***)

st.divider()

val1 = st.text_area("write here your text") 

if st.button("Predict Emotion from text"): 
    if val1 != "":
        pred = model.predict([val1])
        st.success(f'The emotion is: {emotions[pred[0]]}')
    else:
        st.error("Please write a text")

val2 = st.file_uploader('upload here your csv archive ', type='csv') 
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


if st.DownloadButton("Download ", data=pd.DataFrame({'text': ['I am very happy today', 'I am very sad today', 'I am very angry today', 'I am very scared today', 'I am very surprised today']}), file_name='sample.csv', mime='text/csv'):
    st.write("Sample file downloaded")

 




    