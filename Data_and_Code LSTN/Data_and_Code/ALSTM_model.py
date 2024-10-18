# -*-coding:utf-8-*-
import xlrd
import xlwt
import numpy as np
from numpy import *
from sklearn.externals import joblib
import warnings
from keras.models import *
from keras.layers import merge
from keras.layers.core import *
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense,LSTM,MaxPooling1D,Dropout,AveragePooling1D
from keras.layers.convolutional import Conv1D
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from keras.models import load_model

warnings.filterwarnings("ignore")
np.random.seed(7)
train_pw_path='/home/dajinyu/PycharmProjects/dataprocess180528/train_pw_array.pkl'
train_tc_path='/home/dajinyu/PycharmProjects/dataprocess180528/train_tc_array.pkl'

train_pw_list=joblib.load(train_pw_path)
train_tc_list=joblib.load(train_tc_path)

test_pw_path='/home/dajinyu/PycharmProjects/dataprocess180528/test_pw_array.pkl'
test_tc_path='/home/dajinyu/PycharmProjects/dataprocess180528/test_tc_array.pkl'

test_pw_list=joblib.load(test_pw_path)
test_tc_list=joblib.load(test_tc_path)

def save_result(y_test,predicted_values,name):
    w = xlwt.Workbook()
    ws = w.add_sheet("Sheet0")
    sum=0
    s=0
    c=0
    for i in range(y_test.shape[0]):
        ws.write(i,0,float(predicted_values[i][0]))
        ws.write(i, 1, float(y_test[i][0]))
        if y_test[i][0]!=0:
            sum=sum+(abs(predicted_values[i][0]-y_test[i][0])/(y_test[i][0]+1))
            s=s+((predicted_values[i][0]-y_test[i][0])*(predicted_values[i][0]-y_test[i][0]))
            c=c+1
    mape=sum*100/c
    rmse=sqrt(s/c)
    ws.write(0, 5, mape)
    ws.write(0, 6, rmse)

    w.save(name)

def get_dataset(pw_list,tc_list,look_back):
    p=[]
    t=[]
    for j in arange(pw_list.shape[1],step=8):
        p.append(pw_list[0][j])
        t.append(tc_list[0][j])
    p=np.array(p).reshape(1,-1)
    t=np.array(t).reshape(1,-1)
    dataX0, dataX1,dataY = [], [],[]
    for i in range(p.shape[1]-look_back-1):
        a=p[0][i:(i+look_back)]
        b=t[0][i:(i+look_back)]
        dataX0.append(a)
        dataX1.append(b)
        dataY.append(p[0][i+look_back])
    return np.array(dataX0),np.array(dataX1),np.array(dataY)

def normalization(list,max,min):
    temp_list = []
    for i in range(len(list)):
        temp_list = temp_list + list[i]
    temp_array = np.array(temp_list).reshape(-1, 1)
    temp_array=(temp_array-min)/(max-min)
    return temp_array

def LSTM_attention_train(train_pw_list,train_tc_list,epoch,look_back,LR,lstm_units):
    train_pw_list = normalization(train_pw_list, 19979, 0).reshape(1, -1)
    train_tc_list=normalization(train_tc_list,68.8,8.4).reshape(1,-1)
    trainX0,trainX1,trainY=get_dataset(train_pw_list,train_tc_list,look_back)

    samples=trainX0.shape[0]
    times_steps=look_back
    INPUT_DIM=1

    trainX0 = np.reshape(trainX0, (samples, times_steps, 1))

    trainX1 = np.reshape(trainX1, (samples, times_steps, 1))

    ############    model_left  ########
    # input:[samples,times_steps,features]
    inputs_left = Input(shape=(times_steps, INPUT_DIM,),name='inp_left')
    lstm_out_left = LSTM(lstm_units, return_sequences=True, name='lstm0_left')(inputs_left)
    # attention_3d_block(inputs, times_steps)
    a_left = Permute((2, 1), name='permute0_left')(lstm_out_left)
    a_left = Dense(times_steps, activation='softmax', name='dense_left')(a_left)
    a_left_probs = Permute((2, 1), name='permute1_left')(a_left)

    output_attention_mul_left = merge([lstm_out_left, a_left_probs], name='attention_mul_left', mode='mul')
    attention_mul_left = Flatten(name='flatten_left')(output_attention_mul_left)

    ############    model_right  ########
    #  input:[samples,times_steps,features]
    inputs_right = Input(shape=(times_steps, INPUT_DIM,),name='inp_right')
    lstm_out_right = LSTM(lstm_units, return_sequences=True, name='lstm0_right')(inputs_right)
    # attention_3d_block(inputs, times_steps)
    a_right = Permute((2, 1), name='permute0_right')(lstm_out_right)
    a_right = Dense(times_steps, activation='softmax', name='dense_right')(a_right)
    a_right_probs = Permute((2, 1), name='permute1_right')(a_right)

    output_attention_mul_right = merge([lstm_out_right, a_right_probs], name='attention_mul_right', mode='mul')
    attention_mul_right = Flatten(name='flatten_right')(output_attention_mul_right)

    ################ merge  #######
    merge_flatten=merge([attention_mul_left, attention_mul_right], mode='concat',name='merge',concat_axis=1)
    dp = Dropout(0.2, name='dp0')(merge_flatten)
    den=Dense(512, activation='relu', name='dense11')(dp)
    dp1 = Dropout(0.2, name='dp00')(den)
    output = Dense(1, activation='relu', name='dense1')(dp1)
    model = Model(input=[inputs_left,inputs_right], output=output)

    rmsprop = optimizers.RMSprop(LR)
    model.compile(loss="mae", optimizer=rmsprop)
    history = model.fit([trainX0,trainX1], trainY, epochs=epoch, batch_size=64, validation_split=0.05, verbose=1)
    fname='/home/dajinyu/PycharmProjects/dataprocess180528/1h_lstm_attention_result/LSTM+attention-'+str(lstm_units)+'-'+str(look_back)+'-'+str(epoch)+'.h5'
    model.save(fname)

def LSTM_attention_test(test_pw_list,test_tc_list,lstm_units,look_back,epoch):
    test_pw_list = normalization(test_pw_list, 19979, 0).reshape(1, -1)
    test_tc_list = normalization(test_tc_list, 68.8, 8.4).reshape(1, -1)

    testX0, testX1, testY = get_dataset(test_pw_list, test_tc_list, look_back)

    times_steps = look_back

    testX0 = np.reshape(testX0, (testX0.shape[0], times_steps, 1,))
    testX1 = np.reshape(testX1, (testX1.shape[0], times_steps, 1,))

    fname = '/home/dajinyu/PycharmProjects/dataprocess180528/1h_lstm_attention_result/LSTM+attention-' + str(lstm_units) + '-' + str(look_back) + '-' + str(epoch) + '.h5'
    model = load_model(fname)

    testPredict = model.predict([testX0, testX1])
    # Anti normalization
    testPredict = testPredict * (19979 - 0) + 0
    testY = testY * (19979 - 0) + 0
    testY = np.reshape(testY, (-1, 1))

    ename= '/home/dajinyu/PycharmProjects/dataprocess180528/1h_lstm_attention_result/LSTM+attention-' + str(lstm_units) + '-' + str(look_back) + '-' + str(epoch) + '.xls'
    save_result(testY, testPredict, ename)
    print('finish')


def main():
    LSTM_attention_train(train_pw_list,train_tc_list,epoch=24,look_back=6,LR=0.0005,lstm_units=32)
    LSTM_attention_test(test_pw_list,test_tc_list,lstm_units=32,look_back=lb,epoch=24)

main()