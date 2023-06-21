import os

from pickle import load

import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler for data normalization 
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model 

time_steps = 1

target_variables = ['CO2', 'Methane', 'Nitrous_Oxide', 'CFCs',
                    'Hydrochlorofluorocarbons', 'Hydrofluorocarbons',
                    'Total_Heat_Absorbed_GHG', 'Total_Greenhouse_Gases',
                    'GHG_Increase', '%_Change_GHG', 'Surface_Temperature', 'CO2_Concentration']

# Get the absolute path of the CSV file
data_path = os.path.abspath('clean_data.csv')

# Load the data from the CSV file
data = pd.read_csv(data_path)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[target_variables])


class ModelNN:

    def __init__(self, model, data, scaler, is_tf_model=True):
        self.model = model
        self.data = data
        self.scaler = scaler
        self.is_tf_model = is_tf_model

    def predict(self, input_year):
        predicted_val = []
        if input_year > 2021:
            # Prepare the input data
            input_data = self.data[self.data['Year'] <= 2021]
            input_data_scaled = self.scaler.transform(input_data[target_variables])
            # Prepare the input sequence
            input_sequences = []
            for i in range(len(input_data_scaled) - time_steps, len(input_data_scaled)):
                input_sequences.append(input_data_scaled[i - time_steps:i])

            for year in range(2022, input_year + 1):
                input_sequences_array = np.array([sequence for sequence in input_sequences if len(sequence) == time_steps])
                input_sequence_reshaped = np.reshape(input_sequences_array, (input_sequences_array.shape[0], input_sequences_array.shape[1], len(target_variables)))

                # Make predictions for the input sequence
                if self.is_tf_model:
                    predictions = self.model.predict(input_sequence_reshaped[-1].reshape(1, time_steps, len(target_variables)))
                else:
                    predictions = self.model.predict(input_sequence_reshaped[-1].reshape(1, time_steps, len(target_variables)).reshape(1, 12))

                # Inverse transform the predicted values
                prediction = self.scaler.inverse_transform(predictions)

                # Get the predicted values for the current year
                predicted_values = prediction[0]

                # Append the predicted values to the results
                predicted_val.append(predicted_values)

                # Update the input sequence for the next iteration
                input_sequences[0][:-1] = input_sequences[0][1:]  # Remove the first value
                input_sequences[0][-1] = np.array(predictions)  # Add predictions at the last index
        else:
            input_data = self.data[self.data['Year'] < input_year]
            input_data_scaled = self.scaler.transform(input_data[target_variables])
            # Prepare the input sequence
            input_sequence = []
            for i in range(len(input_data_scaled) - time_steps, len(input_data_scaled)):
                input_sequence.append(input_data_scaled[i - time_steps:i])

            input_sequence = np.array(input_sequence)

            # Reshape the input sequence for LSTM (input_shape: [samples, time_steps, features])
            if self.is_tf_model:
                input_sequence = np.reshape(input_sequence, (input_sequence.shape[0], input_sequence.shape[1], len(target_variables)))
            else:
                input_sequence = input_sequence.reshape(1, 12)

            # Make predictions for the input sequence
            predictions = self.model.predict(input_sequence)

            # Inverse transform the predicted values
            predicted_val = self.scaler.inverse_transform(predictions)
        return predicted_val[-1]


class ModelLSRM:
    def __init__(self, coeff: np.ndarray, degree: int = 3):
        self.coeff = coeff
        self.degree = degree

    def predict(self, year: int):
        all_predictions = []
        for c in range(12):
            prediction = 0
            coeff = self.coeff[c]
            for index2 in range(self.degree+1):
                prediction+=coeff[len(coeff)-index2-1]*float(year)**index2
            all_predictions.append(prediction)
        return all_predictions


class ModelARIMA:
    def __init__(self, data, start_year=1979, train_year=2012):
        self.data = data
        self.start_year = start_year
        self.train_year = train_year

    def predict(self, end_year):
        train_data = self.data.loc[self.data['Year']<self.train_year]
        predictions = []
        start = self.start_year - 1979 
        end = end_year - 1979
        arima_order_dict = {
            "CO2": (4, 1, 2),
            "Methane": (0, 2, 2),
            "Nitrous_Oxide": (6, 1, 2),
            "CFCs": (10, 1, 0),
            "Hydrochlorofluorocarbons": (1, 1, 1),
            "Hydrofluorocarbons":  (4, 2, 1),
            "Total_Heat_Absorbed_GHG": (1, 1, 1), #(10, 0, 1),
            "Total_Greenhouse_Gases": (1, 1, 1), #(4, 0, 1),
            "GHG_Increase":  (1, 1, 1),#(6, 0, 1),
            "%_Change_GHG": (8, 0, 1),
            "Surface_Temperature": (1, 1, 0),
            "CO2_Concentration": (1, 1, 1)
        }
        for target in target_variables:
            model = ARIMA(train_data[target], order=arima_order_dict[target])
            model_fit = model.fit()
            y_predict = model_fit.predict(strat=start, end=end)[end]
            predictions.append(y_predict)
        return predictions
    

# Load the trained models

def create_lstm():
    model = load_model('Lstm.h5')  # Load the ANN model
    return ModelNN(model, data, scaler)

def create_ann():
    model = load_model('ann_model.h5')  # Load the ANN model
    return ModelNN(model, data, scaler)

def create_mlp_regressor():
    model  = load(open('mlp_regressor.pickle', "rb"))
    return ModelNN(model, data, scaler, is_tf_model=False)

def create_lsrm():
    coeff = load(open('coefficients.pkl', 'rb'))
    return ModelLSRM(coeff)

def create_arima():
    return ModelARIMA(data)