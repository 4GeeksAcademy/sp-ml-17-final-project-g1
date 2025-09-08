import streamlit as st
import pandas as pd
import transformers
from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline

model = RobertaForSequenceClassification.from_pretrained("../models/RoBERTa")
tokenizer = RobertaTokenizer.from_pretrained("../models/RoBERTa")
emotion_classifier = pipeline("text-classification", 
                              model=model, 
                              tokenizer=tokenizer, 
                              return_all_scores=True)


def emotion_categorizer(text, precission = 80):
    result = emotion_classifier(text)
    for i in range(5):
        if result[0][i]['score'] > precission / 100:
            return result[0][i]['label']
    return 'LABEL_5'

st.title(":brain: Emotion Reader")

st.balloons()


emotions = {'LABEL_2' : 'joy 😄',
            'LABEL_3' : 'sadness 😭',
            'LABEL_0' : 'anger 😡',
            'LABEL_1' : 'fear 😨',
            'LABEL_4' : 'surprise 😲',
            'LABEL_5' : 'The emotion from the text is not clear'}

image_emotion = {'LABEL_2' : 'joy',
                 'LABEL_3' : 'sadness',
                 'LABEL_0' : 'anger',
                 'LABEL_1' : 'fear',
                 'LABEL_4' : 'surprise',
                 'LABEL_5' : '404'}


st.markdown('Powered by :[Karen Paiva Leon](https://github.com/infokaren20), [Nahuel vazquez](https://github.com/najuvgz) y [Adam Candalija Naranjo](https://github.com/AdamCN10)')
st.divider()
st.text('This model predicts emotions from text and csv with only a text column.')
st.image('../docs/asets/emotion-wheel.png')
st.text('This model only can preditct the following emotions: Joy, Sadness, Fear, Anger and Surprise.')

slider = st.slider("how many precission do you want in your emotion prediction? in %", 50, 100, 85)

val1 = st.text_area("write here your text. you can try with: I am sad ") 
if st.button("Predict Emotion from text"):
    if val1.strip():
        st.success(f'The emotion is: {emotions[emotion_categorizer(val1, slider)]}')
        st.image(f'../docs/asets/{image_emotion[emotion_categorizer(val1, slider)]}.webp')
    else:
        st.error("Please write a text")

val2 = st.file_uploader('upload here your csv archive ', type='csv') 
csv = None
if st.button("Predict Emotion from csv"):
    if val2 is not None:
        data = pd.read_csv(val2, names=['text'])
        data['emotion'] = ''
        for i in range(len(data)):
            data['emotion'].iloc[i] = emotions[emotion_categorizer(data['text'].iloc[i])]
        st.success("Emotions predicted successfully")
        st.dataframe(data)
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download data as CSV", data=csv, file_name='predicted_emotions.csv', mime='text/csv')
    else:
        st.error("Please upload a CSV with only a 'text' column")
