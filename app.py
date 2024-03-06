import numpy as np
import pandas as pd
import streamlit as st 
import pickle

model = pickle.load(open('model.pkl', 'rb'))
model_a = pickle.load(open('model_a.pkl', 'rb'))
cols=['combined']    
  
def main(): 
    st.title("Knowledge Management")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    upload_file = st.file_uploader('Upload your csv file')
    
    combined_text = st.text_input("Combined Text"," ")  
    
    if st.button("Predict Category"): 
        data = {'combined':combined_text}
        df=pd.DataFrame(data.values(), columns=['combined'])
            
        features_list = df['combined']  
        prediction = model.predict(features_list)
    
        output = prediction[0]

        st.success('Predicted Category is {}'.format(output))

    if st.button("Predict Area"): 
        data = {'combined':combined_text}
        df=pd.DataFrame(data.values(), columns=['combined'])
            
        features_list = df['combined']  
        prediction = model_a.predict(features_list)
    
        output = prediction[0]

        st.success('Predicted Area is {}'.format(output))



      
if __name__=='__main__': 
    main() 
    
