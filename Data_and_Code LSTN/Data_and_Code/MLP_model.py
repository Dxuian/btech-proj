import xlrd
import xlwt
import numpy as np
from numpy import *
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor

np.random.seed(7)
train_pw_path='/home/dajinyu/PycharmProjects/dataprocess180528/train_pw_array.pkl'
train_tc_path='/home/dajinyu/PycharmProjects/dataprocess180528/train_tc_array.pkl'

train_pw_list=joblib.load(train_pw_path)
train_tc_list=joblib.load(train_tc_path)

test_pw_path='/home/dajinyu/PycharmProjects/dataprocess180528/test_pw_array.pkl'
test_tc_path='/home/dajinyu/PycharmProjects/dataprocess180528/test_tc_array.pkl'

test_pw_list=joblib.load(test_pw_path)
test_tc_list=joblib.load(test_tc_path)

def get_dataset(pw_list,tc_list,look_back):
    p=[]
    t=[]
    for j in arange(pw_list.shape[1],step=8):
        p.append(pw_list[0][j])
        t.append(tc_list[0][j])
    p=np.array(p).reshape(1,-1)
    t=np.array(t).reshape(1,-1)

    dataX,dataY = [], []
    for i in range(p.shape[1]-look_back-1):
        a=p[0][i:(i+look_back)].tolist()
        b=t[0][i:(i+look_back)].tolist()
        dataX.append(a + b)
        dataY.append(p[0][i+look_back])

    return np.array(dataX),np.array(dataY)


def normalization(list,max,min):
    temp_list = []
    for i in range(len(list)):
        temp_list = temp_list + list[i]
    temp_array = np.array(temp_list).reshape(-1, 1)
    temp_array=(temp_array-min)/(max-min)
    return temp_array

def mlp_train(train_pw_list,train_tc_list,look_back):
    train_pw_list = normalization(train_pw_list, 19979, 0).reshape(1, -1)
    train_tc_list = normalization(train_tc_list, 68.8, 8.4).reshape(1, -1)
    trainX, trainY = get_dataset(train_pw_list, train_tc_list, look_back)

    mlp = MLPRegressor(hidden_layer_sizes=(64,16), activation='logistic', solver='adam', alpha=0.0001,
                       batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5,
                       max_iter=800, shuffle=True, random_state=1, tol=0.00001, verbose=1,
                       momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                       validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    mlp.fit(trainX, trainY)
    name='/home/dajinyu/PycharmProjects/dataprocess180528/1h_mlp_result/mlp-'+str(look_back)+'.pkl'
    joblib.dump(mlp, name)

def mlp_test(test_pw_list,test_tc_list,look_back):
    w = xlwt.Workbook()
    ws = w.add_sheet("Sheet0")

    test_tc_list = normalization(test_tc_list, 68.8, 8.4).reshape(1, -1)
    test_pw_list = normalization(test_pw_list, 19979, 0).reshape(1, -1)

    testX, testY = get_dataset(test_pw_list, test_tc_list, look_back)

    name='/home/dajinyu/PycharmProjects/dataprocess180528/1h_mlp_result/mlp-'+str(look_back)+'.pkl'

    mlp=joblib.load(name)
    expect = mlp.predict(testX) * (19979 - 0) + 0
    testY = testY * (19979 - 0) + 0
    sumerr = 0
    se = 0
    c=0
    for i in range(len(expect)):
        ws.write(i, 0, expect[i])
        ws.write(i, 1, testY[i])
        if testY[i]!=0:
            sumerr += abs(expect[i] - testY[i]) / (testY[i] + 1)
            se += (expect[i] - testY[i]) * (expect[i] - testY[i])
            c=c+1


    MAPE = sumerr * 100 / c
    RMSE = sqrt(se / c)

    ws.write(0, 5, 'MAPE')
    ws.write(1, 5, MAPE)
    ws.write(0, 6, 'RMSE')

    ws.write(1, 6, RMSE)

    ename='/home/dajinyu/PycharmProjects/dataprocess180528/1h_mlp_result/mlp-'+str(look_back)+'.xls'
    w.save(ename)
    print('finish')

mlp_train(train_pw_list,train_tc_list,look_back=6)
mlp_test(test_pw_list,test_tc_list,look_back=6)