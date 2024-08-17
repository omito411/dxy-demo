#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.layers import LSTM
from keras.api.layers import Dropout

#######################
# Load data

training_data = pd.read_csv('DXY_training_data.csv')
training_data = training_data.dropna(axis = 0, how ='any')  
training_data = training_data.iloc[:, 4].values  ##this will store the data in a numpy array ##NN needs to be an array

scaler = MinMaxScaler() ###because the data is linearly correlated -- makes it easier for the model to manage 
training_data = scaler.fit_transform(training_data.reshape(-1, 1)) ##application of scaler to the data, the scaler range is -1 to 1
##you don't do normal x and y train because it collects random data, the data needs to be in a sequential order

x_training_data = []##independent variable

y_training_data =[]##dependent
###loop through put one day into dependent variable then 40 will be added as independent variable.

for i in range(40, len(training_data)):
    x_training_data.append(training_data[i-40:i, 0])##40 prior to where i am and taking being stored to x
    y_training_data.append(training_data[i, 0])### y training data
##put both lists into a numpy array
x_training_data = np.array(x_training_data)

y_training_data = np.array(y_training_data)
print(x_training_data.shape)##40 days

print(y_training_data.shape)##y is just one feature so it contains rows of data ##each of the values  ##you have to use the x train and y train to predict
(273, 40)
(273,)
##need to add an additional dimension because this is required for tensor flow

x_training_data = np.reshape(x_training_data, (x_training_data.shape[0],

                                               x_training_data.shape[1],

                                               1))


rnn = Sequential() ##initialising/naming rnn
rnn.add(LSTM(units = 40, return_sequences = True, input_shape = (x_training_data.shape[1], 1)))###no activation function because LSTM defaults to an activation function
##input neurons 40 which is one for each day.
##return sequence true is what makes it recurrent
##increase to 45 to increase the dimesionality
rnn.add(Dropout(0.2))##the droput layer drops some data to help with overfitting and the exploding gradient issue
rnn.add(LSTM(units = 45, return_sequences = True))

rnn.add(Dropout(0.2))

rnn.add(LSTM(units = 45, return_sequences = True))

rnn.add(Dropout(0.2))

rnn.add(LSTM(units = 45))

rnn.add(Dropout(0.2))###final layer never uses the return sequence that makes it recurrent

##activation function is default is tanh
rnn.add(Dense(units = 1)) ###specifies the number of required outputs -- which is the next day stock price

##activation function is default is tanh
rnn.add(Dense(units = 1)) ###specifies the number of required outputs -- which is the next day stock price
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
test_data = pd.read_csv('DXY_test_data.csv')###next 20 days
test_data = test_data.dropna(axis = 0, how ='any') 
test_data = test_data.iloc[:, 4].values

unscaled_training_data = pd.read_csv('DXY_training_data.csv')

unscaled_test_data = pd.read_csv('DXY_test_data.csv')
unscaled_training_data = unscaled_training_data.dropna(axis = 0, how ='any') 
unscaled_test_data = unscaled_test_data.dropna(axis = 0, how ='any') 
all_data=pd.concat((unscaled_training_data['Close'],unscaled_test_data['Close']), axis = 0)
x_test_data = all_data[len(all_data) - len(test_data) - 40:].values


x_test_data = np.reshape(x_test_data, (-1, 1))
final_x_test_data = []

for i in range(40, len(x_test_data)):###index 40 is observation 41 because we start at 0

    final_x_test_data.append(x_test_data[i-40:i, 0])##41st variable and saving in the y variable

final_x_test_data = np.array(final_x_test_data)
final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0], final_x_test_data.shape[1],1))
predictions = rnn.predict(final_x_test_data)

# Page configuration

unscaled_predictions = scaler.inverse_transform(predictions)##you have to unscale data that you-ve scaleed using the inverse transformation


st.write("The last trading day price prediction is for April 2024 is:", unscaled_predictions[0])
st.write("The real last trading day price for May 2024 is:", test_data[0])

alt.themes.enable("dark")

#######################

array = np.array([
    [105.36588],
    [105.365326],
    [105.45252],
    [105.54209],
    [105.51691],
    [105.35042],
    [105.11987],
    [104.954414],
    [104.90948],
    [104.91264],
    [104.925354],
    [104.915085],
    [104.85022],
    [104.66175],
    [104.414024],
    [104.184814],
    [104.037025],
    [103.99133],
    [104.05812],
    [104.213326]
])

# Define the array
array = np.array([
    [105.36588],
    [105.365326],
    [105.45252],
    [105.54209],
    [105.51691],
    [105.35042],
    [105.11987],
    [104.954414],
    [104.90948],
    [104.91264],
    [104.925354],
    [104.915085],
    [104.85022],
    [104.66175],
    [104.414024],
    [104.184814],
    [104.037025],
    [103.99133],
    [104.05812],
    [104.213326]
])

#######################
# Load data
# Create a DataFrame with days as index
days = np.arange(1, len(array) + 1)
df = pd.DataFrame(array, columns=["value"], index=days)
df.index.name = 'day' 

st.write("Enter Day Index:")

if ("plot" not in st.session_state):
    st.session_state.submitted = False

if (st.button("Plot Graph") or st.session_state.submitted):
    st.session_state.submitted = True
    fig = px.scatter(df, x=df.index, y='value', title='DXY Data Points Over Days')
    fig.update_layout(
    xaxis_title='Day', yaxis_title='DXY price')
    st.plotly_chart(fig)

day_input = st.number_input("Enter a day")

if ("submitted" not in st.session_state):
    st.session_state.submitted = False

if (st.button("submit-button") or st.session_state.submitted) and day_input == 0 or day_input < 0 or day_input < 1 or day_input > len(df):
    st.session_state.submitted = True
    st.error('Please enter a valid day index.', icon="🚨")
else:
    st.session_state.submitted = True
    value = df.loc[day_input, 'value']
    st.success(f'The value for day {day_input} is {value}')
