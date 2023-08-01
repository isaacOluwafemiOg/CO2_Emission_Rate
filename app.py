import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image

imag=Image.open('co2_imageai.jpg')

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


np.random.seed(3)
@st.cache_resource(allow_output_mutation=True)
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

@st.cache_resource(allow_output_mutation=True)
def predict_test(pdata):
    model =pickle.load(open('co2_predictor.pkl','rb'))
    pred = model.predict(pdata)
    
    return (pred)
    
    


def main():
    #st.sidebar.image(imag,width=80)#,use_column_width=True)
    st.sidebar.image(imag,use_column_width=True)
    mode =st.sidebar.selectbox('Prediction mode:',['Default Test','Single Prediction','Batch Prediction'],index=0)
    st.sidebar.markdown('''
    ---
    Created by [Isaac Oluwafemi Ogunniyi](https://linkedin.com/in/isaac-oluwafemi-ogunniyi)
    ''')
    
    
    st.title('CO2 Emission Predictor')
    st.write('This web application is towards Net Zero journey with an emphasis on helping SMEs assess\
    the environmental impact of their operations especially with regard to transport. It works by taking input\
    on the features of vehicles and an average monthly distance covered by the vehicle')
    st.write('A machine learning model predicts the CO2 emission rate of the vehicle based on the input vehicle features\
    and subsequently the CO2 emission per month using the input average monthly distance covered')
    st.write('There are three modes of this application and it can be toggled using the drop-down on the left side-bar')

   
    
    with st.spinner("Unpacking the model... Please wait."):
        X_test,test,model,trainscore,testscore = train_model()
    
    col1, col2 = st.columns(2,gap='medium')
    
    col1.metric('Train RMSE:', str(round((trainscore) ** 0.5,3))+' g/km',help='Performance of model on train data')
    col2.metric('Test RMSE:', str(round((testscore) ** 0.5,3))+' g/km',help='Performance of model on test data')

    transmissi = [('A','Automatic'),('AM','Automated Manual'),
                  ('AV','Continuously Variable'),('M','Manual'),
                  ('AS','Automatic with Select Shift')]

    fuelty = [('X','Regular Gasoline'),('Z','Premium Gasoline'),
              ('D','Diesel'),('E','Ethanol(E85)'),('N','Natural gas')]

    rand_distance = np.round(np.random.uniform(low=1877.62,high=2145.74,size=(test.shape[0],)) ,2)
    

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
        
        randistance = pd.DataFrame({'Average Monthly Distance(km)':rand_distance})
        for_view = pd.concat([for_view,randistance],axis=1)

        for_view
        

        if st.button("Predict on unseen test data above"):   
            with st.spinner("Predicting... Please wait."):
                predic = predict_test(for_test)

            st.subheader('Results')
            pred_view = pd.DataFrame(predic)
            
            pred_view.columns = ['Predicted CO2 emission rate in g/km']
            
            emission = np.dot(randistance.T,pred_view)
            pred_view
            st.write('Predicted monthly CO2 emission for the fleet of vehicles above is ' + str(round(np.squeeze(emission)*0.001,3)) + ' kg')
            
        
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
        
        distance = st.number_input('What is the average monthly distance covered by this vehicle in kilometers',
                                   min_value=1.0,value=2011.7,step=0.1)

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
            st.write('The predicted monthly CO2 emission for this vehicle is ' + str(round(predic[0]*distance*0.001,3)) + ' kg')
    


    if mode == 'Batch Prediction':
        st.header('Predicting on Uploaded File')

        st.write('This mode requires you to upload a csv file containing the features of a number of vehicles on which you want to predict\
                 CO2 emissions')
        st.write('Below is a table containing a guide on how the table should be structured')
        
        test_copy = test.drop('CO2 Emissions(g/km)',axis=1)
        test_copy['Average Monthly Distance(km)'] = rand_distance
        col_name = ['Make','Vehicle Class','Engine Size(L)','Cylinders','Transmission','Fuel Type','Fuel Consumption Comb (L/100 km)','number_of_gears',
                    'Average Monthly Distance(km)']
        data_type = list(np.squeeze(test_copy.dtypes))
        data_example = [str(list(np.squeeze(test_copy[i].unique()))) for i in test_copy.columns]
        
        illust = pd.DataFrame({'Column Name':col_name,'Data Type':data_type,'Examples':data_example})
        
        ranges = [2,3,6,7]

        for q in ranges:
            illust.iloc[q,2] = 'Between ' + str(test_copy.iloc[:,q].min()) + ' and ' + str(test_copy.iloc[:,q].max())
        illust.iloc[8,2] = 'float value such as 1910.74'
        illust['Data Type'] = np.where(illust['Data Type']=='object','String',illust['Data Type'])

        illust
        st.write('Warning!!!')
        st.write('Uploading data with wrong data types will lead to unexpected dangerous results')

        u_data = st.file_uploader('Kindly provide a csv file containing the vehicles on which you want to predict',type='csv')
        
        if u_data is not None:
            data = pd.read_csv(u_data)

            st.subheader('Dataset Preview')
            data_ordered = data[col_name]
            st.write('Below is the data you uploaded')
            data_ordered
            
            for i,j in transmissi:
                    data_ordered['Transmission'] = np.where(data_ordered['Transmission']==i,
                                                        j,data_ordered['Transmission'])
            for i,j in fuelty:
                data_ordered['Fuel Type'] = np.where(data_ordered['Fuel Type']==i,
                                                    j,data_ordered['Fuel Type'])
               
            
            
            
            
            if st.button("Predict on uploaded data"):
                
                with st.spinner("Predicting... Please wait."):
                    predic = predict_test(data_ordered)

                st.subheader('Results')
                bpred_view = pd.DataFrame(predic)
                
                bpred_view.columns = ['Predicted CO2 emission rate in g/km']
                
                inp_distance = data_ordered['Average Monthly Distance(km)']
                bemission = np.dot(inp_distance.T,bpred_view)
                bpred_view
                st.write('Predicted monthly CO2 emission for the fleet of vehicles uploaded is ' + str(round(np.squeeze(bemission)*0.001,3)) + ' kg')


 

        else:
            st.write('No dataset Uploaded')


    
    
        





if __name__ == '__main__':
    main()
