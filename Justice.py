import numpy as np
import pandas as pd
import streamlit as st
import joblib


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




def predict_judgment(judgment):

    #input=np.array([[judgment]]).astype(np.str)
    prediction=model.predict([[judgment]])
    return (prediction)

def main():
    st.title("القضايا في المحاكم السعودية")
    judgment = st.text_area("ادخل القضية :")
    
  
    if st.button("Predict"):      
        output=predict_judgment(judgment)
        st.write('the text Predictis ', output)

if __name__== '__main__':
    main()

