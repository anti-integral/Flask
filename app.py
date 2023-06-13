from flask import Flask, request, jsonify, render_template  # Import necessary Flask modules
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler for data normalization
from tensorflow.keras.models import load_model  # Import load_model from Keras for loading trained models
from pickle import load
import os  # Import os for file path operations
import json  # Import json for working with JSON data

app = Flask(__name__)  # Create an instance of the Flask application

# Load the trained models
model_ann = load_model('ann_model.h5')  # Load the ANN model
model_lstm = load_model('Lstm.h5')  # Load the LSTM model

# Get the absolute path of the CSV file
data_path = os.path.abspath('clean_data.csv')

# Load the data from the CSV file
data = pd.read_csv(data_path)

# Define the target variables
target_variables = ['CO2', 'Methane', 'Nitrous_Oxide', 'CFCs',
                    'Hydrochlorofluorocarbons', 'Hydrofluorocarbons',
                    'Total_Heat_Absorbed_GHG', 'Total_Greenhouse_Gases',
                    'GHG_Increase', '%_Change_GHG', 'Surface_Temperature', 'CO2_Concentration']

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[target_variables])

# Define the number of time steps (lags) to consider
time_steps = 1


class LSRM:
    def __init__(self, coeff: np.ndarray, degree: int = 3):
        self.coeff = coeff
        self.degree = degree

    def convert_watt_per_m2_to_joule_per_year(self, watt_per_m2):
        # Constants
        total_surface_area_earth_m2 = 5.1 * 10**14  # Total surface area of the Earth in m²
        seconds_per_year = 60 * 60 * 24 * 365  # Number of seconds in a year
        # Conversion
        joules_per_year = watt_per_m2 * total_surface_area_earth_m2 * seconds_per_year
        return joules_per_year

    def predict(self, year: int):
        all_predictions = []
        for c in range(12):
            prediction = 0
            coeff = self.coeff[c]
            for index2 in range(self.degree+1):
                prediction+=coeff[len(coeff)-index2-1]*float(year)**index2
                # converted_prediction = self.convert_watt_per_m2_to_joule_per_year(prediction)
            all_predictions.append(prediction)
        return all_predictions
    

def create_lsrm():
    coeff = load(open('coefficients.pkl', 'rb'))
    return LSRM(coeff)
    

@app.route('/')
def index():
    return render_template('index.html')  # Render the 'index.html' template


def predict_tf_models(model, input_year):
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
    return predicted_val[-1]


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
    elif model_type == 'polynomial':
        model = create_lsrm()
    else:
        return jsonify({'error': 'Invalid model type, choose between "ann"m "lstm" and "polynomial".'})

    # Rest of the code for prediction
    if model_type in ('ann', 'lstm'):
        predicted_val = predict_tf_models(model, input_year)
    elif model_type == 'polynomial':
        predicted_val = model.predict(input_year)
    else:
        return jsonify({'error': 'Invalid model type, choose between "ann"m "lstm" and "polynomial".'})

    # Prepare the response
    prediction_dict = {}

    def format_number(value):
        if value == 0:
            return "0"
        power = int(np.floor(np.log10(abs(value))))
        coefficient = value / (10 ** power)
        if abs(power) > 5:
            coefficient_str = "{:.3e}".format(coefficient * (10 ** power))
            return coefficient_str.replace("e+", "x10^").replace("e-", "x10^-")
        else:
            return "{:.3f}".format(value)


    # Variables that require conversion to "joules per year"
    variables_to_convert = ["CO2", "Methane", "Nitrous_Oxide", "CFCs", "Hydrochlorofluorocarbons", "Hydrofluorocarbons",
                            "Total_Greenhouse_Gases"]

    # Units for the variables
    units = ["joules per year", "joules per year", "joules per year", "joules per year", "joules per year",
             "joules per year", "ppm", "ppm", "AGGI", "% change per year", "ºC", "ppm"]

    for variable, value in zip(target_variables, predicted_val):
        if variable in variables_to_convert:
            converted_value = value * 5.1e14 * 60 * 60 * 24 * 365
            formatted_number = format_number(converted_value)
            prediction_dict[variable] = formatted_number + " " + "joules per year"
        else:
            unit_index = target_variables.index(variable)
            formatted_number = format_number(value)
            prediction_dict[variable] = formatted_number + " " + units[unit_index]

    # Return the predictions as JSON response
    return jsonify(prediction_dict)

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask application if the script is executed directly