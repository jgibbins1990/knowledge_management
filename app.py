import numpy as np
import pandas as pd
import streamlit as st 
import pickle

model = pickle.load(open('model.pkl', 'rb'))
model_a = pickle.load(open('model_a.pkl', 'rb'))  
  
def main(): 
    st.title("Knowledge Management")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    upload_file = st.file_uploader('Upload your csv file')

    if st.button("View uploaded file"): 
          df = pd.read_csv(upload_file)
          st.dataframe(df, width=2500, height= 300)

    if st.button("Predict Category Area and Topics"): 

          df = pd.read_csv(upload_file)
          cols = ['Title','Summary','Change']
          df['combined'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)),axis=1)
          
          features_list = df['combined']  
          predictions = model.predict(features_list)
          predictions_a = model_a.predict(features_list)
          df['Predicted Category'] = predictions
          df['Predicted Area'] = predictions_a
          
          topics = ['ballot visa youth guidance', 'sponsor register license student', 'sponsor register license worker', 
                                         'document guidance supporting', 'visa late application update' ]
          df['Predicted Topics'] = topics
          st.dataframe(df[['Predicted Category','Predicted Area','Predicted Topics','Title','Change','Summary','Date','Link']], width=2500, height= 300)



if __name__=='__main__': 
    main() 
    
