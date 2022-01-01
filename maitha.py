import streamlit as st
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
import  string

pipe_lr = joblib.load(open('Ministry_Justice.pkl','rb'))

def clean_text(text):
    # stop = nltk.corpus.stopwords('arabic')
    # word_st =('الحمد','لله','تعالي','التوفيق','وصلي','الله','وسلم','علي','نبينا','محمد','اله','وصحبه','اجمعين','عضو','رئيس','الدائره','المحكمه','المحاكم','الحمد','والصلاه','والسلام','القضيه','رقم','لعام','سجل','القاضي','الموافق','المدعي','وكيل','الوكاله','المحاماه','المحاكم','بن','محكمة')
    # for i in word_st:
    #     stop.append(i)

    # word_st2 ='لمدعيه','للمدعي','هويه','جلسه','موكله','الجلسة', 'عليها','رسول','ﷲ','و','العقد','الحكم','الدعوي','للمحاكم','منطوقه','منطوق','رقم','هوية','الهوية','موكلتي','وطنية','بتاريخ','تاريخ','في'
    # for i in word_st2:
    #     stop.append(i)

    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub(r'\s', ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
   

    text = re.sub(r'\s*[A-Za-z]+\b', ' ', text)
    text = re.sub('-', ' ', text)
    text = re.sub('\u200c', ' ', text)
    text = re.sub('\u0640', ' ', text)
    text = re.sub('\u064E', ' ', text)
    text = re.sub('\u0650', ' ', text)

      #Removing Punctuations
   #  arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|"!”…“–ـ'''
   #  text = text.split()
   # text = [w for w in text if not w in stop]
   # text = " ".join(text)
    punctuation=string.punctuation
    text = [word for word in text if word not in punctuation]
    text = ''.join(text)

#      #Removing stopwords
#     text = text.split()
#     text = [w for w in text if not w in stop]
#     text = " ".join(text)

    # Return a list of words
    return text



def predict_judgment(text):
    prediction=pipe_lr.predict([text])
    return prediction[0]

def main():
    st.title("القضايا في المحاكم السعودية")
    judgment = st.text_area("ادخل القضية :")
    Text = clean_text(judgment)

   
  
    if st.button("Predict"):      
        output=predict_judgment(Text)
        st.write('القضية تقع ضمن اختصاص المحكمة :  ', output)

if __name__=='__main__':
    main()

