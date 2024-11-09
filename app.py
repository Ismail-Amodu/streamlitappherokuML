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

def predict(input_data, prediction_type):
    """
    Predicts energy consumption or CO2 emission based on user input.

    Args:
        input_data: A tuple containing the input features.
        prediction_type: "Energy Consumption" or "CO2 Emission".

    Returns:
        A string containing the prediction and recommendation.
    """

    input_array = np.array(input_data).reshape(1, -1)
    model = best_energy_model if prediction_type == "Energy Consumption" else best_co2_model
    importance_df = energy_importance_df if prediction_type == "Energy Consumption" else co2_importance_df

    prediction = model.predict(input_array)[0]

    recommendation = ""
	threshold = get_threshold(prediction_type)
    if prediction >= model_threshold(prediction_type):  # Define threshold function based on model
        recommendation = f"Recommendation: To reduce {prediction_type.lower()}, consider the following:\n"
        for index, row in importance_df.iterrows():
            feature_name = row['Feature']
            feature_value = input_data[index]
            if row['Importance'] > 0.1:
                if feature_value > getattr(energy_kiln_data, feature_name).mean() if prediction_type == "Energy Consumption" else getattr(co2_kiln_data, feature_name).mean():
                    recommendation += f"- Reduce {feature_name} (current value: {feature_value:.2f})\n"
                else:
                    recommendation += f"- Increase {feature_name} (current value: {feature_value:.2f})\n"
    else:
        recommendation = f"The input fields give a {prediction_type.lower()} value. Proceed with the Kiln process."
		
def get_threshold(prediction_type):
    if prediction_type == "Energy Consumption":
        return 1050  # Adjust threshold as needed
    elif prediction_type == "CO2 Emission":
        return 700  # Adjust threshold as needed
    else:
        raise ValueError("Invalid prediction type")


# Streamlit App Interface
st.title("Cement Kiln Prediction System")

prediction_type = st.selectbox("Select Prediction Type", options=["Energy Consumption", "CO2 Emission"])

if prediction_type == "Energy Consumption":
    Power = st.number_input("Power (Kw)", min_value=4000, max_value=6000)
    Power_Consumption = st.number_input("Power Consumption (MWH)", min_value=300000, max_value=500000)
    Power_Consumed = st.number_input("Power Consumed (KWt/h)", min_value=4000, max_value=6000)
    Absorbed_Power = st.number_input("Absorbed Power (Kw)", min_value=3500, max_value=5500)
    Power_Consumption_Max = st.number_input("Power Consumption (Kw) Max.", min_value=6500, max_value=8000)
    Power_rotary_drive = st.number_input("Power (rotary drive) Kw", min_value=500, max_value=800)

    user_input = (Power, Power_Consumption, Power_Consumed, Absorbed_Power, Power_Consumption_Max, Power_rotary_drive)

elif prediction_type == "CO2 Emission":
    Kiln_Temperature = st.number_input("Kiln Temperature (°C)", min_value=1325, max_value=1375)
    Fuel_Consumption = st.number_input("Fuel Consumption (kg/h)", min_value=175, max_value=225)
    Raw_Meal_Composition = st.number_input("Raw Meal Composition (% Limestone))", min_value=82, max_value=88)
    Clinker_Production_Rate = st.number_input("Clinker Production Rate (tons/h)", min_value=80, max_value=110)
    Oxygen_Levels = st.number_input("Oxygen Levels (%)", min_value=3.2, max_value=4.2)
    Kiln_Pressure = st.number_input("Kiln Pressure (bar)", min_value=13.5, max_value=16.5)
    Air_Flow_Rate = st.number_input("Air Flow Rate (m³/h)", min_value=1750, max_value=2250)
    Fuel = st.number_input("Fuel", min_value=0, max_value=1)

    user_input = (Kiln_Temperature, Fuel_Consumption, Raw_Meal_Composition, Clinker_Production_Rate, Oxygen_Levels, Kiln_Pressure, Air_Flow_Rate, Fuel)

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