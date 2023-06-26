import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image

#imag=Image.open('dataware.jpg')

@st.cache(allow_output_mutation=True)
def train_model():
    test = pd.read_csv('test.csv')
    data = pd.read_csv('train.csv')
    model =pickle.load(open('co2_predictor.pkl','rb'))
    
    X = data.drop('CO2 Emissions(g/km)',axis=1)
    y = data['CO2 Emissions(g/km)']
    X_train,X_test,y_train,y_test= train_test_split(X,
    y,test_size=0.3,random_state=2)
    tr_pred = model.predict(X_train)
    te_pred = model.predict(X_test)
    train_score = mean_squared_error(y_train,tr_pred)
    test_score = mean_squared_error(y_test,te_pred)
    return (X_test,test,model,train_score,test_score)

@st.cache(allow_output_mutation=True)
def predict_test(pdata):
    model =pickle.load(open('co2_predictor.pkl','rb'))
    pred = model.predict(pdata)
    
    return (pred)
    
    


def main():
    #st.sidebar.image(imag,width=80)#,use_column_width=True)
    
    mode =st.sidebar.selectbox('Prediction mode:',['Default Test','Single Prediction','Batch Prediction'],index=0)
    st.sidebar.markdown('''
    ---
    Created by [Isaac Oluwafemi Ogunniyi](https://linkedin.com/in/isaac-oluwafemi-ogunniyi)
    ''')
    
    
    st.title('CO2 Emission Predictor')
    
    with st.spinner("Unpacking the model... Please wait."):
        X_test,test,model,trainscore,testscore = train_model()
    st.write('Train RMS:', (trainscore) ** 0.5)
    st.write('Test RMS:', (testscore) ** 0.5)


    if mode == 'Default Test':
        st.header('Predicting Default Test Data')
        st.subheader('Dataset Preview')
        
        for_view = test.drop('CO2 Emissions(g/km)',axis=1)
        for_view

        if st.button("Predict on unseen test data above"):   
            with st.spinner("Predicting... Please wait."):
                predic = predict_test(for_view)

            st.subheader('Results')
            pred_view = pd.DataFrame(predic)
            
            pred_view.columns = ['Prediction']
            
            pred_view
            
        
        st.write('''***''')


    if mode == 'Single Prediction':
        st.header('Predicting on a single input')

        st.sidebar.select_slider

        inp = pd.DataFrame()

        if st.button("Predict on data input"):
            with st.spinner("Predicting... Please wait."):
                predic = predict_test(inp)

            st.subheader('Results')
            pred_view = pd.DataFrame(predic)
            
            pred_view.columns = ['Prediction']
            
            pred_view['Prediction'] = np.where(pred_view['Prediction']==0,'Genuine','Fraud')
            
            pred_view
    


    if mode == 'Batch Prediction':
        st.header('Predicting on Uploaded File')
        u_data = st.file_uploader('The transactions on which you want to predict',type='csv')
        if u_data is not None:
            data = pd.read_csv(u_data,index_col = 'ID')

            st.subheader('Dataset Preview')
            data

            if st.button("Predict on uploaded data"):
                
                with st.spinner("Predicting... Please wait."):
                    predic = predict_test(data)

                st.subheader('Results')
                pred_view = pd.DataFrame(predic)
                
                pred_view.columns = ['Prediction']
                
                pred_view

 

        else:
            st.write('No dataset Uploaded')


    
    
        





if __name__ == '__main__':
    main()
