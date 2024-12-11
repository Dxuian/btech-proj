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
import tensorflow as tf
import holidays
import pandas as pd
# import r2_score from sklearn.metrics
from sklearn.metrics import r2_score
load_data = pd.read_excel('C:\\Users\death\\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\LoadTexasERCOT.xlsx')
weather_data1 = pd.read_excel('C:\\Users\\death\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\WeatherStation1.xlsx')
weather_data2 = pd.read_excel('C:\\Users\\death\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\WeatherStation2.xlsx')
weather_data3 = pd.read_excel('C:\\Users\\death\Desktop\\btech proj\\DATA SET USED IN THE CONFERENCE PAPER TITLED SHORT-TERM LOAD FORECASTING USING AN LSTM NEURAL NETWORK\\DataSet\\WeatherStation3.xlsx')
load_data.drop(columns=[col for col in load_data.columns if col not in ['Hour_End', 'ERCOT']], inplace=True)
weather_data1['DateTime'] = pd.to_datetime(weather_data1[['Year', 'Month', 'Day', 'Hour' ]])
weather_data1.drop(columns=['Year', 'Month', 'Day', 'Hour' ], inplace=True)
weather_data3['DateTime'] = pd.to_datetime(weather_data2[['Year', 'Month', 'Day', 'Hour']])
weather_data3.drop(columns=['Year', 'Month', 'Day', 'Hour'], inplace=True)
weather_data2['DateTime'] = pd.to_datetime(weather_data2[['Year', 'Month', 'Day', 'Hour']])
weather_data2.drop(columns=['Year', 'Month', 'Day', 'Hour'], inplace=True)
merged_data = pd.merge(load_data, weather_data1, left_on='Hour_End', right_on='DateTime')
# print how many rows have Pressure_w3 = Pressure_w1
merged_data2 = pd.merge(load_data, weather_data3, left_on='Hour_End', right_on='DateTime', suffixes=('_w1', '_w3'))
merged_data3 = pd.merge(load_data, weather_data2, left_on='Hour_End', right_on='DateTime', suffixes=('_w1', '_w3', '_w2'))
print(f"Number of rows where Pressure_w3 = Pressure_w1: {merged_data[merged_data['Pressure_w3'] == merged_data['Pressure_w1']].shape[0]}")
merged_data.drop(columns=['DateTime_w1',  'DateTime_w3','Minute_w1','Minute_w3'], inplace=True)
merged_data['TimeOfDayIndex'] = merged_data['Hour_End'].dt.hour
merged_data['DayOfWeekIndex'] = merged_data['Hour_End'].dt.dayofweek
# print('the shape is ' + str(merged_data.head(1)))
print('The head of the merged data is:\n' + merged_data.columns)
print( merged_data.shape)

us_holidays = holidays.US(years=merged_data['Hour_End'].dt.year.unique(), state='TX')
merged_data['HolidayFlag'] = merged_data['Hour_End'].apply(lambda x: 1 if x in us_holidays else 0)
merged_data = pd.get_dummies(merged_data, columns=['TimeOfDayIndex', 'DayOfWeekIndex', 'HolidayFlag'], drop_first=True)
pd.set_option('display.max_columns', None)
if 'DateTime' in merged_data.columns:
    merged_data.drop(['DateTime'], axis=1, inplace=True)
holiday_flag_1_count = merged_data['HolidayFlag_1'].sum()
holiday_flag_0_count = len(merged_data) - holiday_flag_1_count
print(f"Number of rows where HolidayFlag_1 is 1: {holiday_flag_1_count}")
print(f"Number of rows where HolidayFlag_1 is 0: {holiday_flag_0_count}")
print('The shape is ' + str(merged_data.shape))


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(merged_data.drop(columns=['Hour_End']))
scaled_data = pd.DataFrame(scaled_features, columns=merged_data.columns.drop(['Hour_End']))
print(scaled_data.shape)
scaled_data = scaled_data.astype(np.float64)
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:i + time_steps].drop(columns=['ERCOT']).values)
        y.append(data.iloc[i + time_steps]['ERCOT'])
    return np.array(X), np.array(y)
time_steps = 18  
X, y = create_sequences(scaled_data, time_steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)

from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(LSTM(55, return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))  
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(merged_data[['ERCOT']])
scaled_train_predictions = (train_predictions)
scaled_test_predictions = (test_predictions)
scaled_y_train = (y_train.reshape(-1, 1))
scaled_y_test = (y_test.reshape(-1, 1))
print("Scaled Train Predictions:", scaled_train_predictions)
print("Scaled Test Predictions:", scaled_test_predictions)
train_mae = mean_absolute_error(scaled_y_train, scaled_train_predictions)
test_mae = mean_absolute_error(scaled_y_test, scaled_test_predictions)
train_rmse = np.sqrt(mean_squared_error(scaled_y_train, scaled_train_predictions))
test_rmse = np.sqrt(mean_squared_error(scaled_y_test, scaled_test_predictions))
train_r2 = r2_score(scaled_y_train, scaled_train_predictions)
test_r2 = r2_score(scaled_y_test, scaled_test_predictions)

print(f'Training MAE: {train_mae}, RMSE: {train_rmse}, R²: {train_r2}')
print(f'Testing MAE: {test_mae}, RMSE: {test_rmse}, R²: {test_r2}')












# plt.figure(figsize=(14, 18))
# plt.subplot(3, 1, 1)
# plt.plot(scaled_y_test, label='Actual Load', color='blue')
# plt.plot(scaled_test_predictions, label='Predicted Load', color='orange')
# plt.title('Actual vs Predicted Load (Testing Set)')
# plt.xlabel('Time')
# plt.ylabel('Load (MW)')
# plt.legend()
# plt.subplot(3, 1, 2)
# plt.plot(scaled_y_test, label='Actual Load', color='blue')
# plt.title('Actual Load (Testing Set)')
# plt.xlabel('Time')
# plt.ylabel('Load (MW)')
# plt.legend()
# plt.subplot(3, 1, 3)
# plt.plot(scaled_test_predictions, label='Predicted Load', color='orange')
# plt.title('Predicted Load (Testing Set)')
# plt.xlabel('Time')
# plt.ylabel('Load (MW)')
# plt.legend()
# plt.tight_layout()
# plt.show()


