import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import streamlit as st
from streamlit_option_menu import option_menu
global final_Ans
data=pd.read_csv("loan_approval_dataset.csv")

new_prediction = []

# scaler=StandardScaler()
lt=LabelEncoder()
data[" self_employed"]=lt.fit_transform(data[" self_employed"])

data.drop(' no_of_dependents',axis=1,inplace=True)
data.drop(' education',axis=1,inplace=True)
data.drop('loan_id',axis=1,inplace=True)
print(data)

x=data.drop(' loan_status',axis=1)
y=data[' loan_status']

x_train ,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=42)
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)

pre=rf.predict(x_test)


print("Prediction",pre)

acc=accuracy_score(y_test,pre)
print("Accuracy:",acc)

with st.sidebar:
    selected=option_menu("Main Menu",
                         ["Home","Dataset","Signup","Form"],
                         icons=["house","pen","clipboard","table"],default_index=0)

if selected=="Home":
    st.title("Welcome to Loan Approval Predictor!")
    st.header("Learn more about our loan process")
    st.write("Sign up for an account")
    st.write("Fill out loan application form")
    st.write("View and download our loan dataset")
    st.write("Check the status of your loan approval")

elif selected=="Dataset":
    st.header("Dataset Used To Approve The Loan")
    data = pd.read_csv("loan_approval_dataset.csv")
    st.dataframe(data)

elif selected=="Signup":
    st.title("Enter Your Personal Details")
    with st.form(key='personal_form'):
        first_name=st.text_input("First Name")
        last_name=st.text_input("Last Name")
        date_of_birth=st.date_input("Date of Birth",min_value=datetime.date(1900, 1, 1), max_value=datetime.date(2024, 12, 31))
        gender=st.selectbox("Gender",["Male","Female","Other"],key="gender")

        email=st.text_input("Email")
        if st.form_submit_button("Submit"):
            if not first_name or not last_name or not date_of_birth or not gender or not email:
                st.warning("Please fill all the details")
            else:
                data={"First Name":[first_name],"Last Name":[last_name],
                      "Date of Birth":[str(date_of_birth)],"Gender":[gender],
                      "Email":[email]}
                df=pd.DataFrame(data)
                df.to_csv("Project_personal_info.csv",mode='a',header=False,index=False)
                st.write("Data saved successfully!")

elif selected=="Form":
    st.title(" Bank Loan Approval Predictor ")
    with st.form(key='Form'):
        new_employment = st.selectbox("Are you employed?", ["Select","Yes","No"])
        new_income = st.text_input("Enter Your Income:")
        new_loan_amount = st.text_input("Enter Loan Amount:")
        new_loan_term = st.text_input("Enter Loan Term:")
        new_cibil_score = st.text_input("Enter Your Cibil Score:")
        new_residential_assets_value = st.text_input("Enter Your residential Assets Value:")
        new_commercial_assets_value = st.text_input("Enter Your commercial Assets Value:")
        new_luxury_assets_value = st.text_input("Enter Your Luxury Assets Value:")
        new_bank_asset_value = st.text_input("Enter Your Bank Assets Value:")
        if st.form_submit_button("Submit"):
            if not new_employment or not new_income or not new_loan_amount or not new_loan_term or not new_cibil_score or not new_residential_assets_value or not new_luxury_assets_value or not new_bank_asset_value:
                st.warning("Please fill all the details")
            else:
                categories = ['No','Yes']
                oe=OrdinalEncoder(categories=[categories])
                new_employment_encoded=oe.fit_transform([[new_employment]])
                new_employment_input=int(new_employment_encoded[0][0])

                new_data = np.array([[new_employment_input, new_income, new_loan_amount, new_loan_term, new_cibil_score,
                               new_residential_assets_value, new_commercial_assets_value, new_luxury_assets_value,
                               new_bank_asset_value]])

                loan_data = pd.DataFrame(
                    [[new_employment, new_income, new_loan_amount, new_loan_term, new_cibil_score,
                            new_residential_assets_value, new_commercial_assets_value, new_luxury_assets_value,
                            new_bank_asset_value]],
                    columns=['self_employed', 'income', 'loan_amount', 'loan_term', 'cibil_score',
                             'residential_assets_value',
                             'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'])

                loan_data.to_csv('Project_loan_info.csv',mode='a' ,header=False ,index=False)

                new_prediction = rf.predict((new_data))


                st.title("Check the status of your loan application")


                st.write("Your Loan Application is",(new_prediction[0]))
               


