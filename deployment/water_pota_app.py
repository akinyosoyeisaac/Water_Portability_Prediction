import streamlit as st
import pandas as pd
import yaml
import pickle as pk
import json


with open("params.yaml") as file:
    config = yaml.safe_load(file)


def loading_model(config):
    with open(config["paths"]["model"], "rb") as file:
        model = pk.load(file)
    return model

model = loading_model(config=config)
    
columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
def main():
    st.title("WATER POTABILITY PREDICTION")
    st.header("Predicting using Form")
    st.subheader("Please fill the form below")
    with st.form(key="water potability features"):
        pH_value = st.number_input(label="pH value", min_value=0, max_value=14)
        Hardness = st.number_input(label="Hardness")
        Solid = st.number_input(label="Total Dissolve Solid")
        Chloramine = st.number_input(label="Chloramines mg/L")
        Sulfate = st.number_input(label="Sulfate mg/L")
        Conductivity = st.number_input(label="Conductivity μS/cm")
        Organic_carbon = st.number_input(label="Organic_carbon mg/L")
        Trihalomethanes = st.number_input(label="Trihalomethanes ppm")
        Turbidity = st.number_input(label="Turbidity NTU")
        
        submitted = st.form_submit_button("Submit form")
    if submitted:
        features = [pH_value, Hardness, Solid, Chloramine, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
        test_df = pd.DataFrame([features], columns=columns)
        predicted_value = model.predict(test_df)
        
        if predicted_value == 1:
            st.success("THE WATER IS POTABLE")
        else:
            st.success("THE WATER IS NOT POTABLE")
            
    st.header("Predicting Using a CSV file")
    st.subheader("Upload your csv file")
    uploadcsvfile = st.file_uploader("test file", type="csv", accept_multiple_files=False)
    if uploadcsvfile is not None:
        test_df = pd.read_csv(uploadcsvfile)
        pred = model.predict(test_df)
        pred_series = pd.Series(pred, index=test_df.index)
        pred_series = pred_series.map({0:"The water is not potable", 1: "The water is potable"})
        if st.button("Generate prediction"):
            st.dataframe(pred_series)
    
    st.header("MODEL PERFORMANCE")
    visual = st.radio(label="Model Performance", options=["Metrics", "Visuals"])
    
    if visual == "Metrics":
        with open(config["paths"]["metrics"]) as file:
            metrics = json.load(file)["metrics"]
        st.text(metrics) 
    elif visual == "Visuals":
        tab1, tab2 = st.tabs(["Confusion Matrix".upper(), "ROCAUC CURVE"])
        with tab1:
            st.subheader("Confusion Matrix Using Randomforest Classifier")
            st.image(config["paths"]["confusion_matrix"])
        with tab2:
            st.subheader("ROCAUC Curve Using Randomforest Classifier")
            st.image(config["paths"]["roc_curve"])
            
def sidebar():
    with st.sidebar:
        st.subheader("Contributors, Github Link")
        st.write("[Mariam CL](https//github.com/mariam-cl)")
        st.markdown("""
                    1. <a href="https//github.com/akinyosoyeisaac"> Ogunjinmi Isaac </a> 
                    2. [Erica Konadu Antwi](https//github.com/ericakonadu)
                    3. [Mariam CL](https//github.com/mariam-cl)
                    4. [Selasi Ayittah](https//github.com/Selasi3)
                    5. [Tcharrison](tcharrisson)
                    """, unsafe_allow_html=True)
        
        st.subheader("Project Link")
        st.markdown("[Project Link](https//github.com/akinyosoyeisaac/Water_Portability_Prediction)")
        st.subheader("ABOUT")
        st.markdown("""
                    Is project is key as there is an increasing non-availability of potable water esp. in most African country hence, this project is on a projection to building a model that can predict whether consumable water is potable or not for human consumption
                    """)
    
    
            
if __name__ == "__main__":
    sidebar()
    main()