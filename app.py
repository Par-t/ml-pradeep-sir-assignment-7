import pickle

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')

cv=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


def pp(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for i in text:
        if i not in stop_words and i not in punc:
            y.append(i)

    text = y[:]
    y.clear()

    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("SMS Spam Classifier")

input_sms=st.text_input("Enter The message")

preprocessed_sms= pp(input_sms)

vector_input=cv.transform([preprocessed_sms])

result= model.predict(vector_input)[0]

if result==1:
    st.header("SPAM")

else:
    st.header("Not Spam")