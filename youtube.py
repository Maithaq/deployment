import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('modelyoutube.pkl','rb'))
       
                   
def predict_fareamount(views,likes,dislikes,comment_count):
    input=np.array([[views,likes,dislikes,comment_count]]).astype(np.float64)
    prediction=model.predict(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)
def main():
    
    st.title(" ypotube")
    views = st.number_input("views :")
    likes = st.number_input('likes : ')
    dislikes= st.number_input('dislikes :')
    comment_count=st.number_input('comment_count : ')
    
  
    if st.button("Predict"):      
        output=predict_fareamount(views,likes,dislikes,comment_count)
        st.write('youtube ', output)
if __name__=='__main__':
    main()

