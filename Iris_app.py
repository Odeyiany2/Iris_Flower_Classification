import pandas as pd
import joblib
import time
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Iris Flower Classification", page_icon="⚜️")

st.markdown("# ⚜️ Iris Flower Classifier")
st.markdown("##### This web app predicts the specie of an Iris flower based on the measurements given to it.")

#loading our model
model = joblib.load("Iris_Classification_Model.joblib")

option = st.selectbox(
    "Select and option",
    ("Enter measurements manual", "A csv file"))

if option == "Enter measurements manual":
    Sepal_Length = st.text_area("Enter the sepal length")
    Sepal_Width = st.text_area("Enter the sepal width")
    Petal_Length = st.text_area("Enter the petal length")
    Petal_Width = st.text_area("Enter the petal width")

    #creating a dataframe of the manually entered data
    manual_data = pd.DataFrame({
        "Sepal_length": [Sepal_Length],
        "Sepal_width": [Sepal_Width],
        "Petal_length": [Petal_Length],
        "Petal_width": [Petal_Width]
    })
    #function to encode the predicted values
    def iris_analyze(x):
        if x == 0:
            return "Iris-setosa"
        elif x == 1:
            return "Iris-versicolor"
        else:
            return "Iris-virginica"
        
    if st.button("Predict"):
        start = time.time()
        #scaling the features
        scaler = StandardScaler()
        manual_data_scaled = scaler.fit_transform(manual_data)
        prediction = model.predict(manual_data_scaled)
        predicted = iris_analyze(prediction)
        end = time.time()
        st.write("Prediction time taken: ", round(end-start, 2), "seconds.")
        st.write("Prediction score class is: ", prediction)
        st.write("The Predicted Iris Specie is: ", predicted)

elif option=="A csv file":
    st.markdown("#### Your csv file should only contain the sepal length, sepal width, petal length and petal width of the flower.")
    file = st.file_uploader("Upload a csv file")
    #function to encode the predicted values
    def iris_analyze(x):
        if x == 0:
            return "Iris-setosa"
        elif x == 1:
            return "Iris-versicolor"
        else:
            return "Iris-virginica"
    if st.button("Predict"):
        start = time.time()
        data = pd.read_csv(file)
        #scale the features
        scaler_2 = StandardScaler()
        data_scaled = scaler_2.fit_transform(data)
        data["Prediction_score"] = model.predict(data_scaled)
        data["Iris_Specie"] = data["Prediction_score"].apply(iris_analyze)
        end = time.time()
        st.write(data.head())

        #download button for the analyzed csv file
        @st.cache_data 
        def convert_df(df):
            return df.to_csv().encode("utf-8")
        csv = convert_df(data)
        
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Iris Specie Classification.csv',
            mime='text/csv',
        )

else:
    st.write("Nothing was selected.")


















st.divider()
st.markdown("##### Built by Miriam Itopa Odeyiany as a Project for ECX 4.0")
st.write("Copyright © Miriam Itopa Odeyiany, Nigeria 2024")




