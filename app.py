from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import json

app = Flask(__name__)

# Load the trained models
model_ann = load_model('ann_model.h5')
model_lstm = load_model('Lstm.h5')

# Get the absolute path of the CSV file
data_path = os.path.abspath('clean_data.csv')

# Load the data from the CSV file
data = pd.read_csv(data_path)

# Define the target variables
target_variables = ['CO2', 'Methane', 'Nitrous_Oxide', 'CFCs',
       'Hydrochlorofluorocarbons', 'Hydrofluorocarbons',
       'Total_Greenhouse_Gases', 'Total_Greenhouse_Gases_Scaled',
       '1990_Equals_1', 'Change', 'Surface_Temperature', 'CO2_Mean']

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[target_variables])

# Define the number of time steps (lags) to consider
time_steps = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input year and model type from the request
    input_year = int(request.form['year'])
    model_type = request.form['model_type']

    # Select the model based on the user's choice
    if model_type == 'ann':
        model = model_ann
    elif model_type == 'lstm':
        model = model_lstm
    else:
        return jsonify({'error': 'Invalid model type, choose between "ann" and "lstm".'})

    # Rest of the code for prediction
    predicted_val = []
    if input_year > 2021:
        # Prepare the input data
        input_data = data[data['Year'] <= 2021]
        input_data_scaled = scaler.transform(input_data[target_variables])
        # Prepare the input sequence
        input_sequences = []
        for i in range(len(input_data_scaled) - time_steps, len(input_data_scaled)):
            input_sequences.append(input_data_scaled[i - time_steps:i])

        for year in range(2022, input_year + 1):
            input_sequences_array = np.array([sequence for sequence in input_sequences if len(sequence) == time_steps])
            input_sequence_reshaped = np.reshape(input_sequences_array, (input_sequences_array.shape[0], input_sequences_array.shape[1], len(target_variables)))

            # Make predictions for the input sequence
            predictions = model.predict(input_sequence_reshaped[-1].reshape(1, time_steps, len(target_variables)))

            # Inverse transform the predicted values
            prediction = scaler.inverse_transform(predictions)

            # Get the predicted values for the current year
            predicted_values = prediction[0]

            # Append the predicted values to the results
            predicted_val.append(predicted_values)

            # Update the input sequence for the next iteration
            input_sequences[0][:-1] = input_sequences[0][1:]  # Remove the first value
            input_sequences[0][-1] = np.array(predictions)  # Add predictions at the last index
    else:
        input_data = data[data['Year'] < input_year]
        input_data_scaled = scaler.transform(input_data[target_variables])
        # Prepare the input sequence
        input_sequence = []
        for i in range(len(input_data_scaled) - time_steps, len(input_data_scaled)):
            input_sequence.append(input_data_scaled[i - time_steps:i])

        input_sequence = np.array(input_sequence)

        # Reshape the input sequence for LSTM (input_shape: [samples, time_steps, features])
        input_sequence = np.reshape(input_sequence, (input_sequence.shape[0], input_sequence.shape[1], len(target_variables)))

        # Make predictions for the input sequence
        predictions = model.predict(input_sequence)

        # Inverse transform the predicted values
        predicted_val = scaler.inverse_transform(predictions)

    # Prepare the response
    prediction_dict = {}

    def convert_units(value, unit):
        # Constants
        total_surface_area_earth_m2 = 5.1 * 10 ** 14  # Total surface area of the Earth in m²
        seconds_per_year = 60 * 60 * 24 * 365  # Number of seconds in a year
        if unit == "joules per year":
            # Conversion
            joules_per_year = value * total_surface_area_earth_m2 * seconds_per_year
            return joules_per_year
        else:
            return value  # If no conversion is required, return the original value

    # Variables that require conversion to "joules per year"
    variables_to_convert = ["CO2", "Methane", "Nitrous_Oxide", "CFCs", "Hydrochlorofluorocarbons", "Hydrofluorocarbons",
                            "Total_Greenhouse_Gases"]

    # Units for the variables
    units = ["joules per year", "joules per year", "joules per year", "joules per year", "joules per year",
             "joules per year", "joules per year", "ppm", "AGGI", "% change per year", "ºC", "ppm"]

    for variable, value in zip(target_variables, predicted_val[-1]):
        if variable in variables_to_convert:
            converted_value = convert_units(value, "joules per year")
            formatted_number = "{:.3e}".format(converted_value)
            prediction_dict[variable] = formatted_number + " " + "joules per year"
        else:
            unit_index = target_variables.index(variable)
            formatted_number = "{:.3e}".format(value)
            prediction_dict[variable] = formatted_number + " " + units[unit_index]

    # Return the predictions as JSON response
    return jsonify(prediction_dict)

if __name__ == '__main__':
    app.run(debug=True)