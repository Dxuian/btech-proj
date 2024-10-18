# -*-coding:utf-8-*-
import os
import xlrd
import xlwt
import numpy as np
from sklearn.externals import joblib

path='/home/dajinyu/PycharmProjects/dataprocess180528/Data'
files= os.listdir(path)

def process(temppw, temptc,ws0,ws1,rowcount):
    col=len(temppw)
    i=0

    for k in range(col):
        ws0.write(rowcount,k,temppw[k])
        ws1.write(rowcount,k,temptc[k])

    return rowcount+1

w = xlwt.Workbook()
ws0 = w.add_sheet("pw")
ws1 = w.add_sheet("tc")
ws2= w.add_sheet("maxmin")

filecount=0
flagtime='0:0:0'
flagday=1
rowcount=0

pw_max=19979.0 #maximum power
pw_min=1.0  #Minimum power
tc_max=68.8  #maximum module temperature
tc_min=8.4  #minimum module temperature

timepwset=[]
timetcset=[]
for fname in files:

    temppw = []
    temptc = []

    filecount=filecount+1
    print(str(filecount)+fname)
    bk= xlrd.open_workbook(path+'/'+fname)
    shxrange = range(bk.nsheets)
    try:
        sh = bk.sheet_by_name("Sheet0")
    except:
        print("no sheet in %s named Sheet0" % fname)
    nrows = sh.nrows

    date_init= sh.cell_value(1, 0)
    d11=date_init.split('/')
    d22 = d11[2].split()
    d33=d22[1].split(':') #time:hour,minute,second

    start_time=int(d33[0])*60+int(d33[1]) #d33[0]:hour  ,d33[1]:minute

    for i in range(1,nrows - 1):
        if((sh.cell_value(i,16)>19999) or (sh.cell_value(i,19)==0)or(sh.cell_value(i,16)==0)): #Remove outliers
            continue
        date = sh.cell_value(i, 0)
        d1 = date.split('/')
        d2 = d1[2].split()
        d3 = d2[1].split(':')
        month=int(d1[0])
        day = int(d1[1])
        time = str(d2[1])

        if flagday==day:
            if flagtime != time:
                flagtime = time
                temppw.append(sh.cell_value(i, 16))
                temptc.append(sh.cell_value(i, 19))
        else:
            d44=flagtime.split(':')
            end_time=int(d44[0])*60+int(d44[1])
            nzs=int((start_time-60)/8)
            nze=int((11*60-end_time)/8)
            z1=[0]*nzs
            z2=[0]*nze

            d55=time.split(':')
            start_time=int(d55[0])*60+int(d55[1])
            if month==5or month==6 or month==7 or month==8:
                if len(temppw)>87: #if effective value is less than 87, discarded.
                    t1 = [8.4] * nzs
                    t2 = [8.4] * nze
                    temppw=z1+temppw[:-1]+z2
                    temptc=t1+temptc[:-1]+t2
                    timepwset.append(temppw)
                    timetcset.append(temptc)
                    rowcount =process(temppw, temptc,ws0,ws1,rowcount)

            elif month==12 or month==11 or month==1:
                if len(temppw)>62:  #if effective value is less than 62, discarded.
                    t1 = [8.4] * nzs
                    t2 = [8.4] * nze
                    temppw = z1 + temppw[:-1] + z2
                    temptc = t1 + temptc[:-1] + t2
                    timepwset.append(temppw)
                    timetcset.append(temptc)
                    rowcount = process(temppw, temptc, ws0, ws1, rowcount)


            elif month==2 or month==3 or month==10:
                if len(temppw)>77: #if effective value is less than 77, discarded.
                    t1 = [8.4] * nzs
                    t2 = [8.4] * nze
                    temppw = z1 + temppw[:-1] + z2
                    temptc = t1 + temptc[:-1] + t2
                    timepwset.append(temppw)
                    timetcset.append(temptc)
                    rowcount = process(temppw, temptc, ws0, ws1, rowcount)


            elif month==4 or month==9:
                if len(temppw)>82: #if effective value is less than 82, discarded.
                    t1 = [8.4] * nzs
                    t2 = [8.4] * nze
                    temppw = z1 + temppw[:-1] + z2
                    temptc = t1 + temptc[:-1] + t2
                    timepwset.append(temppw)
                    timetcset.append(temptc)
                    rowcount = process(temppw, temptc, ws0, ws1, rowcount)

            flagday = day
            tempdate = []
            temppw = []
            temptc = []
            tempeday = []
            temptarget = []
            i=i-1

ws2.write(0,0,'pw_max')
ws2.write(1,0,pw_max)
ws2.write(0,1,'pw_min')
ws2.write(1,1,pw_min)
ws2.write(0,2,'tc_max')
ws2.write(1,2,tc_max)
ws2.write(0,3,'tc_min')
ws2.write(1,3,tc_min)
ws2.write(0,4,'eday_max')
ws2.write(1,4,eday_max)
ws2.write(0,5,'eday_min')
ws2.write(1,5,eday_min)

newname='timeseries-train.xls'
w.save(newname)
joblib.dump(timepwset, 'train_pw_array.pkl')
joblib.dump(timetcset, 'train_tc_array.pkl')
print(pw_max)
print(pw_min)
print(tc_max)
print(tc_min)
print(eday_max)
print(eday_min)
print('finish')