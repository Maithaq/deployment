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
    congestion_surcharge1 = st.number_input("views :")
    hour = st.number_input('likes : ')
    day= st.number_input('dislikes :')
    PULocation_lon=st.number_input('comment_count : ')
    
  
    if st.button("Predict"):      
        output=predict_fareamount(views,likes,dislikes,comment_count)
        st.write('youtube ', output)
if __name__=='__main__':
    main()

