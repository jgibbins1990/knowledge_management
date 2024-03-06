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
    
    #st.dataframe(df, width=1000, height= 1200)

    if st.button("Predict Category Area and Topics"): 

          df = pd.read_csv(upload_file)
          cols = ['Title','Summary','Change']
          df['combined'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)),axis=1)
          
          features_list = df['combined']  
          predictions = model.predict(features_list)
          predictions_a = model_a.predict(features_list)
          df['Predicted Category'] = predictions
          df['Predicted Area'] = predictions_a

          st.dataframe(df[['Predicted Category','Predicted Area','Title','Change','Summary','Link','Date']], width=2500, height= 1200)

    #     st.download_button("Press to Download Predictions", df, 
    #                        "predictions.csv",
    #                        "text/csv",
    #                        key='download-csv'
    #                       )

    # if st.button("Predict Area"): 
    #     data = {'combined':combined_text}
    #     df=pd.DataFrame(data.values(), columns=['combined'])
            
    #     features_list = df['combined']  
    #     prediction = model_a.predict(features_list)
    
    #     output = prediction[0]

    #     st.success('Predicted Area is {}'.format(output))



      
if __name__=='__main__': 
    main() 
    
