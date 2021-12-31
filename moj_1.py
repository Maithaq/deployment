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


df = pd.read_csv('data_clean')
judgment = df[['judgment_text']]
court_y = df[['court_y']]

judgment_tf  =TfidfVectorizer(min_df=3, max_df=0.9)
svd = TruncatedSVD(n_components = 100)



def main():
    st.title("Ministry of Justice")

    st.write("""### Plese write your judgement text""")

  
    title = st.text_input('Judgemant Text')


    ok = st.button("Show  Judgement Similarities")

     if ok:
        df_judgment_text_new = judgment.append({'judgment_text':title},ignore_index=True)
         ss= judgment_tf.fit_transform(judgment).toarray()
        svdMatrix = svd.fit_transform(ss)
        x = judgment_tf.fit_transform(df_judgment_text_new['judgment_text']).toarray()
        x = svd.fit_transform(x)
        clf = LogisticRegressionCV(cv=5, random_state=0).fit(svdMatrix, y)
        y_pred= clf.predict(x)
if __name__=='__main__':
    main()        
