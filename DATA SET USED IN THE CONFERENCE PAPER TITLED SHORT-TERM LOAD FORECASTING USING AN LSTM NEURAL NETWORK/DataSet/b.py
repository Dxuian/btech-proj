import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the Weather Data
weather_files = ['C:\\Users\\death\\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\WeatherStation1.xlsx',
                'C:\\Users\\death\\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\WeatherStation2.xlsx',
                'C:\\Users\\death\\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\WeatherStation3.xlsx']
weather_data = pd.concat([pd.read_excel(f) for f in weather_files], ignore_index=True)

# Step 2: Prepare Weather Data
# Create a timestamp column
weather_data['Datetime'] = pd.to_datetime(weather_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
weather_data = weather_data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])

# Resample to hourly data if necessary
weather_data = weather_data.resample('H', on='Datetime').mean().reset_index()

# Step 3: Load Electricity Data
load_data = pd.read_excel('C:\\Users\death\\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\LoadTexasERCOT.xlsx')
electricity_data = load_data
electricity_data['Datetime'] = pd.to_datetime(electricity_data['Hour_End'])

# Step 4: Merge Weather and Electricity Data
data_combined = pd.merge(weather_data, electricity_data, on='Datetime', how='inner')

# Step 5: Feature Engineering
# Lag Features
data_combined['Lag1_Load'] = data_combined['ERCOT'].shift(1)
data_combined['Lag2_Load'] = data_combined['ERCOT'].shift(2)

# Time-Based Features
data_combined['Hour'] = data_combined['Datetime'].dt.hour
data_combined['DayOfWeek'] = data_combined['Datetime'].dt.dayofweek
data_combined['IsHoliday'] = data_combined['Datetime'].dt.date.isin([]).astype(int)  # Define holiday dates if needed

# Drop rows with NaN values created by lag features
data_combined = data_combined.dropna()

# Step 6: Normalize Numerical Features
scaler = MinMaxScaler()
numerical_features = ['Relative Humidity', 'Temperature', 'Pressure', 'ERCOT', 'Lag1_Load', 'Lag2_Load']
data_combined[numerical_features] = scaler.fit_transform(data_combined[numerical_features])

# Step 7: Train-Test Split
train = data_combined[data_combined['Datetime'] < '2015-01-01']
test = data_combined[data_combined['Datetime'] >= '2015-01-01']

# Drop the Datetime column for model input
train = train.drop(columns=['Datetime'])
test = test.drop(columns=['Datetime'])

# Step 8: Prepare for LSTM Input
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :-1])  # All features except target
        y.append(data[i + time_steps, -1])  # Target variable (e.g., ERCOT)
    return np.array(X), np.array(y)

# Convert to numpy arrays
time_steps = 18
train_array = train.values
test_array = test.values

X_train, y_train = create_sequences(train_array, time_steps)
X_test, y_test = create_sequences(test_array, time_steps)

# Step 9: Save Prepared Data
np.savez('train_data.npz', X_train=X_train, y_train=y_train)
np.savez('test_data.npz', X_test=X_test, y_test=y_test)

print("Data preparation complete. Training and testing data saved.")
