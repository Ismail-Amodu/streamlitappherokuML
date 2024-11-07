import streamlit as st
import numpy as np
import pickle

# Load the saved models
best_energy_model = pickle.load(open('Kiln_energy_consumption_model.sav', 'rb'))
best_co2_model = pickle.load(open('Kiln_CO2_emission_model.sav', 'rb'))

# Load the saved DataFrames
energy_importance_df = pd.read_csv('energy_feature_importance.csv')
energy_kiln_data = pd.read_csv('energy_kiln_data.csv')

co2_importance_df = pd.read_csv('co2_feature_importance.csv')
co2_kiln_data = pd.read_csv('co2_kiln_data.csv')


def predict_energy_consumption(input_data):
    """
    Predicts energy consumption and provides recommendations.

    Args:
        input_data: A tuple containing the input features.

    Returns:
        A string containing the prediction and recommendation.
    """

    input_array = np.array(input_data).reshape(1, -1)
    prediction = best_energy_model.predict(input_array)[0]

    recommendation = ""
    if prediction >= 1050:
        recommendation = "Recommendation: To reduce energy consumption, consider the following:\n"
        for index, row in energy_importance_df.iterrows():
            feature_name = row['Feature']
            feature_value = input_data[index]
            if row['Importance'] > 0.1:
                if feature_value > energy_kiln_data[feature_name].mean():
                    recommendation += f"- Reduce {feature_name} (current value: {feature_value:.2f})\n"
                else:
                    recommendation += f"- Increase {feature_name} (current value: {feature_value:.2f})\n"
    else:
        recommendation = "The input fields give an efficient energy consumption. Proceed with the Kiln process."

    return f"Predicted Energy Consumption: {prediction:.2f} kWh/t-clinker\n{recommendation}"


def predict_co2_emission(input_data):
    """
    Predicts CO2 emission and provides recommendations.

    Args:
        input_data: A tuple containing the input features.

    Returns:
        A string containing the prediction and recommendation.
    """

    input_array = np.array(input_data).reshape(1, -1)
    prediction = best_co2_model.predict(input_array)[0]

    recommendation = ""
    if prediction >= 700:
        recommendation = "Recommendation: To reduce CO2 emission, consider the following:\n"
        for index, row in co2_importance_df.iterrows():
            feature_name = row['Feature']
            feature_value = input_data[index]
            if row['Importance'] > 0.1:  # Threshold for significant features
                if feature_value > co2_kiln_data[feature_name].mean():
                    recommendation += f"- Reduce {feature_name} (current value: {feature_value:.2f})\n"
                else:
                    recommendation += f"- Increase {feature_name} (current value: {feature_value:.2f})\n"
    else:
        recommendation = "The input fields give a reduced CO2 emission. The adjustment enhances sustainability by reducing CO2 emission in cement plant."

    return f"Predicted CO2 Emission: {prediction:.2f} kg/ton-cement\n{recommendation}"
Power (Kw)	Power Consumption (MWH)	Power Consumed (KWt/h)	Absorbed Power (Kw)	Power Consumption (Kw) Max.	Power (rotary drive) Kw

# Streamlit App Interface
st.title("Cement Kiln Prediction System")

# Prediction Type Selection Menu
prediction_type = st.selectbox("Select Prediction Type", options=["Energy Consumption", "CO2 Emission"])
Power (Kw)	Power Consumption (MWH)	Power Consumed (KWt/h)	Absorbed Power (Kw)	Power Consumption (Kw) Max.	Power (rotary drive) Kw

# User Input Fields (Displayed based on selection)
if prediction_type == "Energy Consumption":
	# Input features for Energy Consumption model
    Power = st.number_input("Power (Kw)", min_value=1400, max_value=1600)
    Power_Consumption = st.number_input("Power Consumption (MWH)", min_value=500, max_value=1000)
    Power_Consumed = st.number_input("Power Consumed (KWt/h)", min_value=1400, max_value=1600)
    Absorbed_Power = st.number_input("Absorbed Power (Kw)", min_value=500, max_value=1000)
	Power_Consumption_Max = st.number_input("Power Consumption (Kw) Max.", min_value=1400, max_value=1600)
    Power_rotary_drive = st.number_input("Power (rotary drive) Kw", min_value=500, max_value=1000)
    
    user_input = (Power, Power_Consumption, Power_Consumed, Absorbed_Power, Power_Consumption_Max, Power_rotary_drive)  

elif prediction_type == "CO2 Emission":
    # Input features for CO2 Emission model
    Limestone_Content = st.number_input("Limestone Content (%)", min_value=75, max_value=95)
    Clay_Content = st.number_input("Clay Content (%)", min_value=5, max_value=25)
	Silica_Content = st.number_input("Silica Content (%)", min_value=1, max_value=10)
    Alumina_Content = st.number_input("Alumina Content (%)", min_value=1, max_value=7)
	Iron_Oxide_Content = st.number_input("Iron Oxide Content (%)", min_value=0.5, max_value=5)
    Fuel_Consumption = st.number_input("Fuel Consumption (kg/ton clinker)", min_value=100, max_value=150)
	Electricity_Consumption = st.number_input("Electricity Consumption (kWh/ton clinker)", min_value=80, max_value=120)
	Kiln_Thermal_Efficiency = st.number_input("Kiln Thermal Efficiency (%)", min_value=70, max_value=90)
    Clinker_Ratio = st.number_input("Clinker Ratio (%)", min_value=65, max_value=85)
   
    user_input = (Limestone_Content, Clay_Content, Silica_Content, Alumina_Content, Iron_Oxide_Content, Fuel_Consumption, Electricity_Consumption, Kiln_Thermal_Efficiency, Clinker_Ratio)  

# Make predictions based on the selected prediction type
if prediction_type == "Energy Consumption":
    prediction = predict_energy_consumption(user_input)
    st.subheader("Energy Consumption Prediction:")
    st.write(prediction)
elif prediction_type == "CO2 Emission":
    prediction = predict_co2_emission(user_input)
    st.subheader("CO2 Emission Prediction:")
    st.write(prediction)
else:
    st.write("Please select a prediction type.")