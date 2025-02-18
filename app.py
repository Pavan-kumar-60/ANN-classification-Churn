# Importing all the neccessary libraies.
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model # type: ignore
import pickle
import streamlit as st

# loading ANN model
model = load_model('model.h5')

# loading all the neccessary transformations
with open('label_encoder_gen.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('ohe_encoder.pkl', 'rb') as file:
    OHE_encoder = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)

# streamlit app
st.title('Customer Churn Prediction')

# user input
geography = st.selectbox('Geography', OHE_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18,100)
balance = st.number_input('Balance')
credit_score = st.number_input('CreditScore')
estimated_salary = st.number_input('EstimatedSalary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('IsActiveMember',[0,1])

# creating dict
input_data = {
    'CreditScore': credit_score,
    'Geography': geography ,
    'Gender': label_encoder.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard':has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

# Creating dataframe
input_data = pd.DataFrame(input_data, index=[0])

# converting geography feature into numerical value
encoded_geo = OHE_encoder.transform([input_data['Geography']]).toarray()
encoded_geo = pd.DataFrame(encoded_geo, columns=OHE_encoder.get_feature_names_out())

# concatenating encoded_geo with original dataframe
input_data = pd.concat([input_data, encoded_geo], axis=1)

# dropping geography original feature
input_data.drop(columns='Geography', inplace=True)

# feature scaling
input_data = scalar.transform(input_data)

# Prediction
prediction = model.predict(input_data)
prediction_prob = prediction[0][0]

st.write(f'Probability of churning: {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')


