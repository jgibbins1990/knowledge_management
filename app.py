import numpy as np
import pandas as pd
import streamlit as st 
import pickle

model = pickle.load(open('model.pkl', 'rb'))
encoder_dict = pickle.load(open('encoder.pkl', 'rb')) 
cols=['combined']    
  
def main(): 
    st.title("Category Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Income Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    combined_text = st.text_input("Combine Text"," ")  
    
    if st.button("Predict Category"): 
        features = [['combined']]
        data = {'combined':combined}
        print(data)
        df=pd.DataFrame([list(data.values())], columns=['combined'])
            
        features_list = df.values.tolist()    
        prediction = model.predict(features_list)
    
        output = int(prediction[0])
        text = output

        st.success('Predicted Category is {}'.format(text))
      
if __name__=='__main__': 
    main() 
    
