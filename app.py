from flask import Flask, request, jsonify, render_template  # Import necessary Flask modules
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
from pickle import load
import os  # Import os for file path operations
import json  # Import json for working with JSON data

from ml_models import create_lstm, create_ann, create_mlp_regressor, create_lsrm, create_arima

app = Flask(__name__)  # Create an instance of the Flask application


# Define the target variables
target_variables = ['CO2', 'Methane', 'Nitrous_Oxide', 'CFCs',
                    'Hydrochlorofluorocarbons', 'Hydrofluorocarbons',
                    'Total_Heat_Absorbed_GHG', 'Total_Greenhouse_Gases',
                    'GHG_Increase', '%_Change_GHG', 'Surface_Temperature', 'CO2_Concentration']


# Define the number of time steps (lags) to consider
time_steps = 1
 

@app.route('/')
def index():
    return render_template('index.html')  # Render the 'index.html' template



@app.route('/predict', methods=['POST'])
def predict():
    # Get the input year and model type from the request
    input_year = int(request.form['year'])
    model_type = request.form['model_type']

    # Select the model based on the user's choice
    if model_type == 'ann':
        model = create_ann()
    elif model_type == 'lstm':
        model = create_lstm()
    elif model_type == 'polynomial':
        model = create_lsrm()
    elif model_type == 'arima':
        model = create_arima()
    elif model_type == 'mlp_regressor':
        model = create_mlp_regressor()
    else:
        return jsonify({'error': 'Invalid model type, choose between "ann"m "lstm" and "polynomial".'})

    predicted_val = model.predict(input_year)
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