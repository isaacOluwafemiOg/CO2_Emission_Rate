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
    
    col1, col2 = st.columns(2,gap='medium')
    
    col1.metric('Train RMSE:', str(round((trainscore) ** 0.5,3))+' g/km')
    col2.metric('Test RMSE:', str(round((testscore) ** 0.5,3))+' g/km')

    transmissi = [('A','Automatic'),('AM','Automated Manual'),
                  ('AV','Continuously Variable'),('M','Manual'),
                  ('AS','Automatic with Select Shift')]

    fuelty = [('X','Regular Gasoline'),('Z','Premium Gasoline'),
              ('D','Diesel'),('E','Ethanol(E85)'),('N','Natural gas')]

    if mode == 'Default Test':
        st.header('Predicting on Default Test Data')
        st.subheader('Dataset Preview')
        
        for_view = test.drop('CO2 Emissions(g/km)',axis=1)
        for_test = for_view.copy()
        for i,j in transmissi:
            for_view['Transmission'] = np.where(for_view['Transmission']==i,
                                                j,for_view['Transmission'])
        for i,j in fuelty:
            for_view['Fuel Type'] = np.where(for_view['Fuel Type']==i,
                                                j,for_view['Fuel Type'])
        for_view

        

        if st.button("Predict on unseen test data above"):   
            with st.spinner("Predicting... Please wait."):
                predic = predict_test(for_test)

            st.subheader('Results')
            pred_view = pd.DataFrame(predic)
            
            pred_view.columns = ['Predicted CO2 emission rate in g/km']
            
            pred_view
            
        
        st.write('''***''')


    if mode == 'Single Prediction':
        st.header('Predicting on a single input')
        st.write('Select the features that best describe the vehicle whose emission rate you want to predict')
        
        make_list = ['ACURA', 'ALFA ROMEO', 'ASTON MARTIN', 'AUDI', 'BENTLEY', 'BMW',
       'BUICK', 'CADILLAC', 'CHEVROLET', 'CHRYSLER', 'DODGE', 'FIAT',
       'FORD', 'GMC', 'HONDA', 'HYUNDAI', 'INFINITI', 'JAGUAR', 'JEEP',
       'KIA', 'LAMBORGHINI', 'LAND ROVER', 'LEXUS', 'LINCOLN', 'MASERATI',
       'MAZDA', 'MERCEDES-BENZ', 'MINI', 'MITSUBISHI', 'NISSAN',
       'PORSCHE', 'RAM', 'ROLLS-ROYCE', 'SCION', 'SMART', 'SRT', 'SUBARU',
       'TOYOTA', 'VOLKSWAGEN', 'VOLVO', 'GENESIS', 'BUGATTI']
        
        vclass_list = ['COMPACT', 'SUV - SMALL', 'MID-SIZE', 'TWO-SEATER', 'MINICOMPACT',
       'SUBCOMPACT', 'FULL-SIZE', 'STATION WAGON - SMALL',
       'SUV - STANDARD', 'VAN - CARGO', 'VAN - PASSENGER',
       'PICKUP TRUCK - STANDARD', 'MINIVAN', 'SPECIAL PURPOSE VEHICLE',
       'STATION WAGON - MID-SIZE', 'PICKUP TRUCK - SMALL']

        trans_list = ['Automatic with Select Shift', 'Manual', 'Continuously Variable', 'Automated Manual',
                       'Automatic']

        fuelt_list = ['Regular Gasoline','Premium Gasoline',
              'Diesel','Ethanol(E85)','Natural gas']


        consume = st.slider('Fuel consumption in litres per 100km',4.1,26.1,step=0.1)
        fuel = st.selectbox('Fuel Type:',fuelt_list,index=1)
        gears = st.slider('Number of Gears:',0,10,step=1)
        transmission = st.selectbox('Transmission Type:',trans_list,index=1)
        engine = st.slider('Engine size in Litres:',0.9,8.4,step=0.1)
        v_class = st.selectbox('Class of Vehicle:',vclass_list,index=2)
        
        cylinder = st.slider('Number of Engine Cylinders:',3,16,step=1)
        
        make = st.selectbox('Make of vehicle:',make_list,index=0)
        
        inp = pd.DataFrame({'Make':[make],'Vehicle Class':[v_class],'Engine Size(L)':[engine],
                            'Cylinders':[cylinder],'Transmission':[transmission],
                            'Fuel Type':[fuel],'Fuel Consumption Comb (L/100 km)':[consume],
                            'number_of_gears':[gears]})
        inp_view = inp.copy()
        for i,j in transmissi:
            inp['Transmission'] = np.where(inp['Transmission']==j,
                                                i,inp['Transmission'])
        for i,j in fuelty:
            inp['Fuel Type'] = np.where(inp['Fuel Type']==j,
                                                i,inp['Fuel Type'])
        
        st.markdown('''

         ---
        ''')
        
        st.write('Below is the tabular view of your selected features:')
        inp_view

        if st.button("Predict on data input"):
            with st.spinner("Predicting... Please wait."):
                predic = predict_test(inp)

            st.subheader('Results')
            st.write('The predicted CO2 emission rate is ' + str(round(predic[0],3)) + ' g/km')
    


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
