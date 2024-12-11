
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import holidays
import tensorflow as tf
# Load datasets
load_data = pd.read_excel('C:\\Users\death\\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\LoadTexasERCOT.xlsx')
weather_data1 = pd.read_excel('C:\\Users\\death\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\WeatherStation1.xlsx')
weather_data2 = pd.read_excel('C:\\Users\\death\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\WeatherStation2.xlsx')
weather_data3 = pd.read_excel('C:\\Users\\death\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\WeatherStation3.xlsx')
target = 'WEST'
print("Columns in weather_data1:", weather_data1.columns)
weather_data1['DateTime'] = pd.to_datetime(weather_data1[['Year', 'Month', 'Day', 'Hour' ]])
weather_data1.drop(columns=['Year', 'Month', 'Day', 'Hour' ], inplace=True)
weather_data3['DateTime'] = pd.to_datetime(weather_data2[['Year', 'Month', 'Day', 'Hour']])
weather_data3.drop(columns=['Year', 'Month', 'Day', 'Hour'], inplace=True)
weather_data2['DateTime'] = pd.to_datetime(weather_data2[['Year', 'Month', 'Day', 'Hour']])
weather_data2.drop(columns=['Year', 'Month', 'Day', 'Hour'], inplace=True)
load_data.drop(columns=[col for col in load_data.columns if col not in ['Hour_End', target]], inplace=True)
merged_data = pd.merge(load_data, weather_data1, left_on='Hour_End', right_on='DateTime')
merged_data2 = pd.merge(load_data, weather_data3, left_on='Hour_End', right_on='DateTime')
merged_data3 = pd.merge(load_data, weather_data2, left_on='Hour_End', right_on='DateTime' )
final_data = pd.concat([merged_data, merged_data2, merged_data3], axis=0)
merged_data  = final_data
merged_data.drop(columns=['DateTime'], inplace=True)
us_holidays = holidays.US(years=merged_data['Hour_End'].dt.year.unique(), state='TX')
merged_data['HolidayFlag'] = merged_data['Hour_End'].apply(lambda x: 1 if x in us_holidays else 0)
holiday_flag_1_count = merged_data['HolidayFlag'].sum()
holiday_flag_0_count = len(merged_data) - holiday_flag_1_count
print(f"Number of rows where HolidayFlag_1 is 1: {holiday_flag_1_count}")
print(f"Number of rows where HolidayFlag_1 is 0: {holiday_flag_0_count}")
merged_data['TimeOfDayIndex'] = merged_data['Hour_End'].dt.hour
merged_data['DayOfWeekIndex'] = merged_data['Hour_End'].dt.dayofweek
for hour in range(24):
    merged_data[f'TimeOfDay_{hour}'] = (merged_data['TimeOfDayIndex'] == hour).astype(int)
for day in range(7):
    merged_data[f'DayOfWeek_{day}'] = (merged_data['DayOfWeekIndex'] == day).astype(int)
merged_data.drop(columns=['TimeOfDayIndex', 'DayOfWeekIndex'], inplace=True)
if 'DateTime' in merged_data.columns:
    merged_data.drop(['DateTime'], axis=1, inplace=True)
print(merged_data.head())
print('Column names are:\n' + ', '.join(merged_data.columns))
print('shape of the data is:', merged_data.shape)
numerical_features = ['Pressure' , 'Relative Humidity' , 'Temperature' , target  ]
categorical_features = [col for col in merged_data.columns if col not in numerical_features]
scaler = MinMaxScaler()
merged_data[numerical_features] = scaler.fit_transform(merged_data[numerical_features])

# print the  range of each column numerical and categorical min and max 
print('Numerical Features Range:')
print(merged_data[numerical_features].min())
print(merged_data[numerical_features].max())
print('Categorical Features Range:')
print(merged_data[categorical_features].min())
print(merged_data[categorical_features].max())


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(merged_data.drop(columns=['Hour_End' , ]))

# Convert scaled features back to DataFrame and re-include the 'Hour_End' and 'ERCOT' columns
scaled_data = pd.DataFrame(scaled_features, columns=merged_data.columns.drop(['Hour_End'  ]))

scaled_data
print(scaled_data.dtypes)
scaled_data.head(5)
# Ensure all data is of type float64
scaled_data = scaled_data.astype(np.float64)

# Function to create sequences
def create_sequences(data, time_steps, target):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:i + time_steps].drop(columns=[target]).values)
        y.append(data.iloc[i + time_steps][target])
    return np.array(X), np.array(y)

# Assuming scaled_data is already defined and contains the necessary data
time_steps = 18  # Example time steps
X, y = create_sequences(scaled_data, time_steps, target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Ensure all data is of type float64
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)

# Import dropout
from tensorflow.keras.layers import Dropout

# Define the model
model = Sequential()
model.add(LSTM(55, return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer for load prediction

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Create DataFrames for inverse transformation
train_predictions_df = pd.DataFrame(train_predictions, columns=[target])
test_predictions_df = pd.DataFrame(test_predictions, columns=[target])
y_train_df = pd.DataFrame(y_train, columns=[target])
y_test_df = pd.DataFrame(y_test, columns=[target])

# Inverse transform the predictions
scaled_train_predictions = scaler.inverse_transform(pd.concat([scaled_data.drop(columns=[target]).iloc[:len(train_predictions)], train_predictions_df], axis=1))[:, -1]
scaled_test_predictions = scaler.inverse_transform(pd.concat([scaled_data.drop(columns=[target]).iloc[:len(test_predictions)], test_predictions_df], axis=1))[:, -1]
scaled_y_train = scaler.inverse_transform(pd.concat([scaled_data.drop(columns=[target]).iloc[:len(y_train)], y_train_df], axis=1))[:, -1]
scaled_y_test = scaler.inverse_transform(pd.concat([scaled_data.drop(columns=[target]).iloc[:len(y_test)], y_test_df], axis=1))[:, -1]

print("Scaled Train Predictions:", scaled_train_predictions)
print("Scaled Test Predictions:", scaled_test_predictions)
print("Scaled Train Predictions:", scaled_train_predictions)
print("Scaled Test Predictions:", scaled_test_predictions)




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Calculate MAE and RMSE
train_mae = mean_absolute_error(scaled_y_train, scaled_train_predictions)
test_mae = mean_absolute_error(scaled_y_test, scaled_test_predictions)
train_rmse = np.sqrt(mean_squared_error(scaled_y_train, scaled_train_predictions))
test_rmse = np.sqrt(mean_squared_error(scaled_y_test, scaled_test_predictions))

# Calculate R² score
train_r2 = r2_score(scaled_y_train, scaled_train_predictions)
test_r2 = r2_score(scaled_y_test, scaled_test_predictions)

# Print the results
print(f'Training MAE: {train_mae}, RMSE: {train_rmse}, R²: {train_r2}')
print(f'Testing MAE: {test_mae}, RMSE: {test_rmse}, R²: {test_r2}')




# Step 8: Plot the results in separate subplots
plt.figure(figsize=(14, 18))

# Plot overlapping actual and predicted load
plt.subplot(3, 1, 1)
plt.plot(scaled_y_test, label='Actual Load', color='blue')
plt.plot(scaled_test_predictions, label='Predicted Load', color='orange')
plt.title('Actual vs Predicted Load (Testing Set)')
plt.xlabel('Time')
plt.ylabel('Load (MW)')
plt.legend()

# Plot actual load
plt.subplot(3, 1, 2)
plt.plot(scaled_y_test, label='Actual Load', color='blue')
plt.title('Actual Load (Testing Set)')
plt.xlabel('Time')
plt.ylabel('Load (MW)')
plt.legend()

# Plot predicted load
plt.subplot(3, 1, 3)
plt.plot(scaled_test_predictions, label='Predicted Load', color='orange')
plt.title('Predicted Load (Testing Set)')
plt.xlabel('Time')
plt.ylabel('Load (MW)')
plt.legend()

plt.tight_layout()
plt.show()


