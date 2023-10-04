import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('diabetes.csv')
st.title('web deployment of medical diagnostic app')
st.subheader('is the person diabetic?')
st.set_option('deprecation.showPyplotGlobalUse', False)
if st.sidebar.checkbox('View Data',False):
    st.write(df)
if st.sidebar.checkbox('view distribution',False):
    df.hist()
    plt.tight_layout()
    #df.barplot()
    st.pyplot()
rfc=pickle.load(open('model.pkl','rb'))
pregs=st.number_input('Pregnancies',0,17,0)
glu=st.slider('Glucose',44,199,44)
bp=st.slider('BloodPressure',20,140,20)
skin=st.slider('SkinThickness',7,99,7)
ins=st.slider('Insulin',14,850,14)
bmi = st.slider('BMI',18,67,10)
dbp=st.slider('DiabetesPedigreeFunction',0.07,2.8,0.07)
age=st.slider('Age',21,85,21)


if rfc.predict([[pregs,glu,bp,skin,ins,bmi,dbp,age]])[0]==1:
    st.subheader('Diabetic')
else:
    st.subheader('non-diabetic')