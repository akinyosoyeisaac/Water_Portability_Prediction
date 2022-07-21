import streamlit as st
import pandas as pd
import yaml
import pickle as pk


with open("params.yaml") as file:
    config = yaml.safe_load(file)

@st.cache
with open(config["paths"]["model"], "rb") as file:
    model = pk.load(file)
    
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
            st.markdown("<h2>THE WATER IS POTABLE</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2>THE WATER IS NOT POTABLE</h2>", unsafe_allow_html=True)
            
    st.header("Predicting Using a CSV file")
    st.subheader("Upload your csv file")
    uploadcsvfile = st.file_uploader("test file", type="csv", accept_multiple_files=False)
    if uploadcsvfile not None:
        test_df = pd.read_csv(uploadcsvfile)
        pred = model.predict(test_df)
        pred_series = pd.Series(pred, index=test_df.index)
        pred_series = pred_series.map({0:"The water is not potable", 1: "The water is potable"})
        if st.button("Generate prediction"):
            st.dataframe(pred_series)
        
    
    
if __name__ == "__main__":
    main()