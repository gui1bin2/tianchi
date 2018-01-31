# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:03:45 2018

@author: user
"""

import pandas as pd
import numpy as np
import tkinter
from tkinter.filedialog import askopenfilename
from dateutil.parser import parse
#导入原始数据
def selectPath():
    path_ = askopenfilename()
    path.set(path_)
    root.destroy()
root = tkinter.Tk()
path = tkinter.StringVar()
root.title("导入训练集")
root.geometry('300x100')
tkinter.Label(root,text = "原数据路径CSV:").grid(row = 0, column = 0)
tkinter.Entry(root, textvariable = path).grid(row = 0, column = 1)
tkinter.Button(root, text = "路径选择", command = selectPath).grid(row = 0, column = 2)
root.mainloop()
x = path.get()
data = pd.read_csv(x,encoding='gbk',engine='python')

def selectPath():
    path_ = askopenfilename()
    path.set(path_)
    root.destroy()
root = tkinter.Tk()
path = tkinter.StringVar()
root.title("导入测试集")
root.geometry('300x100')
tkinter.Label(root,text = "原数据路径CSV:").grid(row = 0, column = 0)
tkinter.Entry(root, textvariable = path).grid(row = 0, column = 1)
tkinter.Button(root, text = "路径选择", command = selectPath).grid(row = 0, column = 2)
root.mainloop()
x = path.get()
aa = pd.read_csv(x,encoding='gbk',engine='python')


aa['血糖']=answerB
aa['误差'] = aa['血糖']-aa['29A']
aa['误差平方'] =  aa['误差']**2
aa['误差平方'].sum()/2000


testA['血糖'] = answerA
data = pd.concat([data, testA])
data = data.reset_index(drop=False)

df = data[['id','体检日期','性别','年龄','*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶',
       '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积','中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
       '血糖']]

df = df.drop(2953) #天门冬+丙氨最高，血糖低
df = df.drop(2220)
df = df.drop(2186)
df = df.drop(5067)
df = df.drop(5826)
df = df.drop(4160)
df = df.drop(2160) #谷氨过高
df = df.drop(1791) #谷氨过高
df = df.drop(222) #总脂肪肝
'''
c1 = df[df['血糖']>=6.1]
c2 = df[(df['血糖']<6.1)&(df['血糖']>=3.9)]
c3 = df[df['血糖']<3.9]
t = c1.median()
t = pd.DataFrame(t)
t.columns = ['高血糖中位数']
t2 = c2.median()
t['正常血糖中位数'] = pd.DataFrame(t2)
t3 = c3.median()
t['低血糖中位数'] = pd.DataFrame(t3)
m1 = c1.mean()
m2 = c2.mean()
m3 = c3.mean()
t['高血糖平均数'] = pd.DataFrame(m1)
t['正常血糖平均数'] = pd.DataFrame(m2)
t['低血糖平均数'] = pd.DataFrame(m3)
t['中位幅度'] = (t['高血糖中位数']-t['正常血糖中位数'])/t['正常血糖中位数']
t['平均数幅度'] = (t['高血糖平均数']-t['正常血糖平均数'])/t['正常血糖平均数']
t['中位幅度2'] = (t['正常血糖中位数']-t['低血糖中位数'])/t['低血糖中位数']
t['平均数幅度2'] = (t['正常血糖平均数']-t['低血糖平均数'])/t['低血糖平均数']
t['高血糖四分位'] = c1.quantile(0.75) 
'''

df['性别'] = df['性别'].replace(('男','女','??'),('1','0','0')).astype(float)
df['体检日期'] = (pd.to_datetime(df['体检日期']) - parse('2017-10-09')).dt.days
df['空值数量'] = df.isnull().sum(axis=1)
df = df.fillna(df.median())
df.ix[df['空值数量']>0,'是否空值'] = 1
df.ix[df['空值数量']==0,'是否空值'] = 0
df = df[df['甘油三酯']<=30]
df = df[df['血糖']<=30]
df['总胆固醇'] = df['总胆固醇'].astype(float)
df.ix[df['年龄']>=40,'年龄分类'] = 1
df.ix[df['年龄']<40,'年龄分类'] = 0
df.ix[df['*天门冬氨酸氨基转换酶']<=49,'*天门冬氨酸氨基转换酶分类'] = 0
df.ix[df['*天门冬氨酸氨基转换酶分类']!=0,'*天门冬氨酸氨基转换酶分类'] = 1
df.ix[df['*丙氨酸氨基转换酶']<=49,'*丙氨酸氨基转换酶分类'] = 0
df.ix[df['*丙氨酸氨基转换酶分类']!=0,'*丙氨酸氨基转换酶分类'] = 1
df.ix[df['*总蛋白']<=82,'*总蛋白分类'] = 0
df.ix[df['*总蛋白分类']!=0,'*总蛋白分类'] = 1
df.ix[df['白蛋白']<=55,'白蛋白分类'] = 0
df.ix[df['白蛋白分类']!=0,'白蛋白分类'] = 1
df.ix[df['*球蛋白']<=38,'*球蛋白分类'] = 0
df.ix[df['*球蛋白分类']!=0,'*球蛋白分类'] = 1
df.ix[df['*碱性磷酸酶']<=125,'*碱性磷酸酶分类'] = 0
df.ix[df['*碱性磷酸酶分类']!=0,'*碱性磷酸酶分类'] = 1
df.ix[(df['*r-谷氨酰基转换酶']>=3)&(df['*r-谷氨酰基转换酶']<=69),'*r-谷氨酰基转换酶分类'] = 0
df.ix[df['*r-谷氨酰基转换酶分类']!=0,'*r-谷氨酰基转换酶分类'] = 1
df.ix[(df['尿素']>=2.9)&(df['尿素']<=7.1),'尿素分类'] = 0
df.ix[df['尿素分类']!=0,'尿素分类'] = 1
df.ix[(df['肌酐']>=44)&(df['肌酐']<=108),'肌酐分类'] = 0
df.ix[df['肌酐分类']!=0,'肌酐分类'] = 1
df.ix[(df['尿酸']>=125)&(df['尿酸']<=420),'尿酸分类'] = 0
df.ix[df['尿酸分类']!=0,'尿酸分类'] = 1
df.ix[df['甘油三酯']<=1.71,'甘油三酯分类'] = 0
df.ix[df['甘油三酯分类']!=0,'甘油三酯分类'] = 1
df.ix[df['总胆固醇']<=5.7,'总胆固醇分类'] = 0
df.ix[df['总胆固醇分类']!=0,'总胆固醇分类'] = 1
df.ix[df['低密度脂蛋白胆固醇']<=3.7,'低密度脂蛋白胆固醇分类'] = 0
df.ix[df['低密度脂蛋白胆固醇分类']!=0,'低密度脂蛋白胆固醇分类'] = 1
#玄学
df.ix[df['年龄']<=60,'高血糖年龄'] = 0
df.ix[df['高血糖年龄']!=0,'高血糖年龄'] = 1
df.ix[df['甘油三酯']<=2.80,'高血糖甘油三酯'] = 0
df.ix[df['高血糖甘油三酯']!=0,'高血糖甘油三酯'] = 1
df.ix[df['*r-谷氨酰基转换酶']<=58,'高血糖*r-谷氨酰基转换酶'] = 0
df.ix[df['高血糖*r-谷氨酰基转换酶']!=0,'高血糖*r-谷氨酰基转换酶'] = 1
df.ix[df['*丙氨酸氨基转换酶']<=38,'高血糖*丙氨酸氨基转换酶'] = 0
df.ix[df['高血糖*丙氨酸氨基转换酶']!=0,'高血糖*丙氨酸氨基转换酶'] = 1   
df['高血糖系数'] = df['高血糖年龄']+df['高血糖甘油三酯']+df['高血糖*r-谷氨酰基转换酶']+df['高血糖*丙氨酸氨基转换酶']
df['血脂系数'] = df['甘油三酯分类']+df['总胆固醇分类']+df['低密度脂蛋白胆固醇分类']
df['脂肪肝系数'] = df['*天门冬氨酸氨基转换酶分类']+df['*丙氨酸氨基转换酶分类']+df['*r-谷氨酰基转换酶分类']+df['白蛋白分类']+df['*球蛋白分类']
df['危险系数'] = df['年龄分类']+df['*天门冬氨酸氨基转换酶分类']+df['*丙氨酸氨基转换酶分类']+df['*总蛋白分类']+df['白蛋白分类']+df['*球蛋白分类']+df['*碱性磷酸酶分类']+df['*r-谷氨酰基转换酶分类']+df['肌酐分类']+df['尿素分类']+df['尿酸分类']+df['甘油三酯分类']+df['总胆固醇分类']+df['低密度脂蛋白胆固醇分类']
#rank
df['甘油三酯排名'] = df['甘油三酯'].rank()
df['甘油三酯排名'] = df['甘油三酯排名']/df['甘油三酯排名'].max()
df['*r-谷氨酰基转换酶排名'] = df['*r-谷氨酰基转换酶'].rank()
df['*r-谷氨酰基转换酶排名'] = df['*r-谷氨酰基转换酶排名']/df['*r-谷氨酰基转换酶排名'].max()
df['年龄排名'] = df['年龄'].rank()
df['年龄排名'] = df['年龄排名']/df['年龄排名'].max()
df['*丙氨酸氨基转换酶排名'] = df['*丙氨酸氨基转换酶'].rank()
df['*丙氨酸氨基转换酶排名'] = df['*丙氨酸氨基转换酶排名']/df['*丙氨酸氨基转换酶排名'].max()
df['四指标综合排名'] = (df['甘油三酯排名']+df['*r-谷氨酰基转换酶排名']+df['年龄排名']+df['*丙氨酸氨基转换酶排名'])/4
#1.29
df['白细胞计数排名'] = df['白细胞计数'].rank()
df['白细胞计数排名'] = df['白细胞计数排名']/df['白细胞计数排名'].max()
df['低密度脂蛋白胆固醇排名'] = df['低密度脂蛋白胆固醇'].rank()
df['低密度脂蛋白胆固醇排名'] = df['低密度脂蛋白胆固醇排名']/df['低密度脂蛋白胆固醇排名'].max()
df['*碱性磷酸酶排名'] = df['*碱性磷酸酶'].rank()
df['*碱性磷酸酶排名'] = df['*碱性磷酸酶排名']/df['*碱性磷酸酶排名'].max()
df['三指标综合排名'] = (df['白细胞计数排名'] +df['低密度脂蛋白胆固醇排名']+df['*碱性磷酸酶排名'])
df['嗜酸细胞%排名'] = df['嗜酸细胞%'].rank()
df['嗜酸细胞%排名'] = df['嗜酸细胞%排名']/df['嗜酸细胞%排名'].max()
df['嗜碱细胞%排名'] = df['嗜碱细胞%'].rank()
df['嗜碱细胞%排名'] = df['嗜碱细胞%排名']/df['嗜碱细胞%排名'].max()
df['尿酸排名'] = df['尿酸'].rank()
df['尿酸排名'] = df['尿酸排名']/df['尿酸排名'].max()
df['负相关综合排名'] = (df['嗜酸细胞%排名']+df['嗜碱细胞%排名']+df['尿酸排名'])/3

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


df2 = df[['空值数量','是否空值','危险系数','体检日期','性别',
          '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶','高血糖系数','高血糖年龄','高血糖甘油三酯','高血糖*r-谷氨酰基转换酶',
       '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
       '脂肪肝系数','血脂系数','甘油三酯排名','*r-谷氨酰基转换酶排名','年龄排名','*丙氨酸氨基转换酶排名','四指标综合排名',
       '白细胞计数排名','低密度脂蛋白胆固醇排名','*碱性磷酸酶排名','三指标综合排名',
       '嗜酸细胞%排名','嗜碱细胞%排名','尿酸排名','负相关综合排名','血糖']]

x,y = df2.drop(['血糖'],axis=1),df2[['血糖']]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.001,random_state = 10)

#数据标准化
y_train = np.array(y_train)
y_test = np.array(y_test)
#x标准化，#y标准化
ss_x = preprocessing.StandardScaler().fit(x_train)
x_train = ss_x.transform(x_train)
x_test = ss_x.transform(x_test)
ss_y = preprocessing.StandardScaler().fit(y_train)
y_train = ss_y.transform(y_train.reshape(-1, 1)) 
y_test = ss_y.transform(y_test.reshape(-1, 1)) 

alg = RandomForestRegressor(bootstrap=1, criterion='mse', max_depth=60,
           max_features=56, max_leaf_nodes=None,min_impurity_split=1e-07,
           min_samples_leaf=4,min_samples_split=6, min_weight_fraction_leaf=0.0,
           n_estimators=500,n_jobs=1, oob_score=False, random_state=10,
           verbose=0, warm_start=False)
alg.fit(x_train,y_train)
alg_score = alg.score(x_train,y_train)

#还原
result_rf = alg.predict(x_test)
y_restore = result_rf*ss_y.scale_+ss_y.mean_
y_test = pd.DataFrame(y_test*ss_y.scale_+ss_y.mean_)
result_rf = pd.DataFrame(y_restore)
result_rf.columns=['预测血糖']
result_rf['实际血糖'] = y_test.reset_index(drop=True)
result_rf['误差'] = result_rf['实际血糖']-result_rf['预测血糖']
result_rf['误差率'] = result_rf['误差']/result_rf['实际血糖']
gg5 = result_rf[result_rf.误差率.between(-0.05, 0.05)]
gg10 = result_rf[result_rf.误差率.between(-0.1, 0.1)]
print(gg5.误差率.size/result_rf.实际血糖.size)
print(gg10.误差率.size/result_rf.实际血糖.size)
result_rf['误差平方'] =  result_rf['误差']**2
result_rf['误差平方'].sum()/12

'''    
from sklearn.ensemble import GradientBoostingRegressor as GBR
gbr = GBR(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=28, max_features=30,
             max_leaf_nodes=None, min_impurity_split=1e-07,
             min_samples_leaf=3, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=200,
             presort='auto', random_state=1, subsample=1.0, verbose=0,
             warm_start=False)
gbr.fit(x_train,y_train)
gbr_score = gbr.score(x_train,y_train)    
result_rf = gbr.predict(x_test)
y_restore = result_rf*ss_y.scale_+ss_y.mean_
y_test = pd.DataFrame(y_test*ss_y.scale_+ss_y.mean_)
result_rf = pd.DataFrame(y_restore)
result_rf.columns=['预测血糖']
result_rf['实际血糖'] = y_test.reset_index(drop=True)
result_rf['误差'] = result_rf['实际血糖']-result_rf['预测血糖']
result_rf['误差率'] = result_rf['误差']/result_rf['实际血糖']
gg5 = result_rf[result_rf.误差率.between(-0.05, 0.05)]
gg10 = result_rf[result_rf.误差率.between(-0.1, 0.1)]
print(gg5.误差率.size/result_rf.实际血糖.size)
print(gg10.误差率.size/result_rf.实际血糖.size)
result_rf['误差平方'] =  result_rf['误差']**2
result_rf['误差平方'].sum()/542
'''     

         
#测试集
test = testB
test['性别'] = test['性别'].replace(('男','女','??'),('1','0','0')).astype(float)
test['体检日期'] = (pd.to_datetime(test['体检日期']) - parse('2017-10-09')).dt.days
test['空值数量'] = test.isnull().sum(axis=1)
test = test.fillna(test.median())
test.ix[test['空值数量']>0,'是否空值'] = 1
test.ix[test['空值数量']==0,'是否空值'] = 0
test['总胆固醇'] = test['总胆固醇'].astype(float)
test.ix[test['年龄']>=40,'年龄分类'] = 1
test.ix[test['年龄']<40,'年龄分类'] = 0
test.ix[test['*天门冬氨酸氨基转换酶']<=49,'*天门冬氨酸氨基转换酶分类'] = 0
test.ix[test['*天门冬氨酸氨基转换酶分类']!=0,'*天门冬氨酸氨基转换酶分类'] = 1
test.ix[test['*丙氨酸氨基转换酶']<=49,'*丙氨酸氨基转换酶分类'] = 0
test.ix[test['*丙氨酸氨基转换酶分类']!=0,'*丙氨酸氨基转换酶分类'] = 1
test.ix[test['*总蛋白']<=82,'*总蛋白分类'] = 0
test.ix[test['*总蛋白分类']!=0,'*总蛋白分类'] = 1
test.ix[test['白蛋白']<=55,'白蛋白分类'] = 0
test.ix[test['白蛋白分类']!=0,'白蛋白分类'] = 1
test.ix[test['*球蛋白']<=38,'*球蛋白分类'] = 0
test.ix[test['*球蛋白分类']!=0,'*球蛋白分类'] = 1
test.ix[test['*碱性磷酸酶']<=125,'*碱性磷酸酶分类'] = 0
test.ix[test['*碱性磷酸酶分类']!=0,'*碱性磷酸酶分类'] = 1
test.ix[(test['*r-谷氨酰基转换酶']>=3)&(test['*r-谷氨酰基转换酶']<=69),'*r-谷氨酰基转换酶分类'] = 0
test.ix[test['*r-谷氨酰基转换酶分类']!=0,'*r-谷氨酰基转换酶分类'] = 1
test.ix[(test['尿素']>=2.9)&(test['尿素']<=7.1),'尿素分类'] = 0
test.ix[test['尿素分类']!=0,'尿素分类'] = 1
test.ix[(test['肌酐']>=44)&(test['肌酐']<=108),'肌酐分类'] = 0
test.ix[test['肌酐分类']!=0,'肌酐分类'] = 1
test.ix[(test['尿酸']>=125)&(test['尿酸']<=420),'尿酸分类'] = 0
test.ix[test['尿酸分类']!=0,'尿酸分类'] = 1
test.ix[test['甘油三酯']<=1.71,'甘油三酯分类'] = 0
test.ix[test['甘油三酯分类']!=0,'甘油三酯分类'] = 1
test.ix[test['总胆固醇']<=5.7,'总胆固醇分类'] = 0
test.ix[test['总胆固醇分类']!=0,'总胆固醇分类'] = 1
test.ix[test['低密度脂蛋白胆固醇']<=3.7,'低密度脂蛋白胆固醇分类'] = 0
test.ix[test['低密度脂蛋白胆固醇分类']!=0,'低密度脂蛋白胆固醇分类'] = 1
#玄学
test.ix[test['年龄']<=60,'高血糖年龄'] = 0
test.ix[test['高血糖年龄']!=0,'高血糖年龄'] = 1
test.ix[test['甘油三酯']<=2.80,'高血糖甘油三酯'] = 0
test.ix[test['高血糖甘油三酯']!=0,'高血糖甘油三酯'] = 1
test.ix[test['*r-谷氨酰基转换酶']<=58,'高血糖*r-谷氨酰基转换酶'] = 0
test.ix[test['高血糖*r-谷氨酰基转换酶']!=0,'高血糖*r-谷氨酰基转换酶'] = 1
test.ix[test['*丙氨酸氨基转换酶']<=38,'高血糖*丙氨酸氨基转换酶'] = 0
test.ix[test['高血糖*丙氨酸氨基转换酶']!=0,'高血糖*丙氨酸氨基转换酶'] = 1
test['高血糖系数'] = test['高血糖年龄']+test['高血糖甘油三酯']+test['高血糖*r-谷氨酰基转换酶']+test['高血糖*丙氨酸氨基转换酶']
test['血脂系数'] = test['甘油三酯分类']+test['总胆固醇分类']+test['低密度脂蛋白胆固醇分类']
test['脂肪肝系数'] = test['*天门冬氨酸氨基转换酶分类']+test['*丙氨酸氨基转换酶分类']+test['*r-谷氨酰基转换酶分类']+test['白蛋白分类']+test['*球蛋白分类']
test['危险系数'] = test['年龄分类']+test['*天门冬氨酸氨基转换酶分类']+test['*丙氨酸氨基转换酶分类']+test['*总蛋白分类']+test['白蛋白分类']+test['*球蛋白分类']+test['*碱性磷酸酶分类']+test['*r-谷氨酰基转换酶分类']+test['肌酐分类']+test['尿素分类']+test['尿酸分类']+test['甘油三酯分类']+test['总胆固醇分类']+test['低密度脂蛋白胆固醇分类']
#rank
test['甘油三酯排名'] = test['甘油三酯'].rank()
test['甘油三酯排名'] = test['甘油三酯排名']/test['甘油三酯排名'].max()
test['*r-谷氨酰基转换酶排名'] = test['*r-谷氨酰基转换酶'].rank()
test['*r-谷氨酰基转换酶排名'] = test['*r-谷氨酰基转换酶排名']/test['*r-谷氨酰基转换酶排名'].max()
test['年龄排名'] = test['年龄'].rank()
test['年龄排名'] = test['年龄排名']/test['年龄排名'].max()
test['*丙氨酸氨基转换酶排名'] = test['*丙氨酸氨基转换酶'].rank()
test['*丙氨酸氨基转换酶排名'] = test['*丙氨酸氨基转换酶排名']/test['*丙氨酸氨基转换酶排名'].max()
test['四指标综合排名'] = (test['甘油三酯排名']+test['*r-谷氨酰基转换酶排名']+test['年龄排名']+test['*丙氨酸氨基转换酶排名'])/4
#1.29
test['白细胞计数排名'] = test['白细胞计数'].rank()
test['白细胞计数排名'] = test['白细胞计数排名']/test['白细胞计数排名'].max()
test['低密度脂蛋白胆固醇排名'] = test['低密度脂蛋白胆固醇'].rank()
test['低密度脂蛋白胆固醇排名'] = test['低密度脂蛋白胆固醇排名']/test['低密度脂蛋白胆固醇排名'].max()
test['*碱性磷酸酶排名'] = test['*碱性磷酸酶'].rank()
test['*碱性磷酸酶排名'] = test['*碱性磷酸酶排名']/test['*碱性磷酸酶排名'].max()
test['三指标综合排名'] = (test['白细胞计数排名'] +test['低密度脂蛋白胆固醇排名']+test['*碱性磷酸酶排名'])
test['嗜酸细胞%排名'] = test['嗜酸细胞%'].rank()
test['嗜酸细胞%排名'] = test['嗜酸细胞%排名']/test['嗜酸细胞%排名'].max()
test['嗜碱细胞%排名'] = test['嗜碱细胞%'].rank()
test['嗜碱细胞%排名'] = test['嗜碱细胞%排名']/test['嗜碱细胞%排名'].max()
test['尿酸排名'] = test['尿酸'].rank()
test['尿酸排名'] = test['尿酸排名']/test['尿酸排名'].max()
test['负相关综合排名'] = (test['嗜酸细胞%排名']+test['嗜碱细胞%排名']+test['尿酸排名'])/3


test = test[['空值数量','是否空值','危险系数','体检日期','性别',
          '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶','高血糖系数','高血糖年龄','高血糖甘油三酯','高血糖*r-谷氨酰基转换酶',
       '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
       '脂肪肝系数','血脂系数','甘油三酯排名','*r-谷氨酰基转换酶排名','年龄排名','*丙氨酸氨基转换酶排名','四指标综合排名',
       '白细胞计数排名','低密度脂蛋白胆固醇排名','*碱性磷酸酶排名','三指标综合排名',
       '嗜酸细胞%排名','嗜碱细胞%排名','尿酸排名','负相关综合排名']]



p_x = ss_x.transform(test)
result_rfp = alg.predict(p_x)
p_y_restore = result_rfp*ss_y.scale_+ss_y.mean_

result = pd.DataFrame(p_y_restore)
result.to_csv('D:/WinCode/自如数据/血糖预测.csv',header=None,index=False, float_format='%.4f')


#特征权重1
from sklearn.cross_validation import cross_val_score, ShuffleSplit
scores = []
for i in range(x_train.shape[1]):
     score = cross_val_score(alg,x_train[:,i:i+1],y_train,
                             cv = ShuffleSplit(len(x_train),3,.3))
     scores.append((round(np.mean(score), 3),i))
weight_rf = pd.DataFrame(scores)
weight_rf.columns=['特征相对权重', '序号']
weight_rf = weight_rf[[ '序号','特征相对权重']]
weight_rf['特征名称'] = ['是否空值','危险系数','体检日期','性别',
          '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶','高血糖系数','高血糖年龄','高血糖甘油三酯','高血糖*r-谷氨酰基转换酶',
       '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
       '脂肪肝系数','血脂系数','甘油三酯排名','*r-谷氨酰基转换酶排名','年龄排名','*丙氨酸氨基转换酶排名','四指标综合排名']
weight_rf = weight_rf.sort_values(['特征相对权重'],ascending=False)