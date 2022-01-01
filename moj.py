import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.isri import ISRIStemmer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import streamlit as st



model=pickle.load(open('modelgov.pkl','rb'))

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|"!”…“–ـ'''

def remove_punctuations(text):
    translator = str.maketrans(' ', ' ', arabic_punctuations)
    return text.translate(translator)

#judgment=map(remove_punctuations)

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    # text = replace('\w*\d\w*', ' ')
    # text = replace('\\', ' ')
    # text = replace(r'\s', ' ')
    # text = replace(r'\s*[A-Za-z]+\b', ' ')
    # text = replace('-', ' ')
    # text = replace('\u200c', ' ')
    # text = replace('\u0640', '')
    # text = replace('\u064E', '')
    # text = replace('\u0650', '')

    return text

#judgment =map(normalize_arabic)

# def Tfid(text):
#     cv_tfidf = TfidfVectorizer()
#     X_train_tf = cv_tfidf.fit_transform(text)
#     return X_train_tf


# def svd(text):
#     svd = TruncatedSVD(n_components = 100)
#     svdMatrix = svd.fit_transform(text)
#     return svdMatrix




def predict_judgment(judgment):
    input=np.array([[judgment]]).astype(np.str)
    prediction=model.predict(input)
    return (prediction)


def main():
    
    st.title("القضايا في المحاكم السعودية")
    judgment = st.text_input("ادخل القضية :")
    remove_punctuations(judgment)
    normalize_arabic(judgment)
    #judgment = scv_tfidf.fit_transform(judgment)
    #judgment = svd.fit_transform(judgment)
    # Tfid(judgment)
    # svd(judgment)


    if st.button("Predict"):      
        output=predict_judgment(judgment)
        st.write('the text Predictis ', output)

if __name__=='__main__':
    main()

