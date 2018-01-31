# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:07:58 2018

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:03:45 2018

@author: user
"""

import pandas as pd
import numpy as np
import tkinter
from tkinter.filedialog import askopenfilename
import time
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
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
df = pd.read_csv(x,encoding='gbk',engine='python')

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
test = pd.read_csv(x,encoding='gbk',engine='python')

df = df[['id','体检日期','性别','年龄','*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶',
       '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积','中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
       '血糖']]

df = df.drop(2953) #天门冬+丙氨最高，血糖低
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
df['空值数量'] = df.isnull().sum(axis=1)
df = df.fillna(df.median())
df = df[df['甘油三酯']<=30]
df.ix[df['空值数量']>0,'是否空值'] = 1
df.ix[df['空值数量']==0,'是否空值'] = 0
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
df.ix[df['年龄']<=63,'高血糖年龄'] = 0
df.ix[df['高血糖年龄']!=0,'高血糖年龄'] = 1
df.ix[df['甘油三酯']<=2.80,'高血糖甘油三酯'] = 0
df.ix[df['高血糖甘油三酯']!=0,'高血糖甘油三酯'] = 1
df.ix[df['*r-谷氨酰基转换酶']<=58.585,'高血糖*r-谷氨酰基转换酶'] = 0
df.ix[df['高血糖*r-谷氨酰基转换酶']!=0,'高血糖*r-谷氨酰基转换酶'] = 1
df.ix[df['*丙氨酸氨基转换酶']<=38.495,'高血糖*丙氨酸氨基转换酶'] = 0
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



train = df[['是否空值','危险系数','体检日期','id','性别','年龄',
          '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶','高血糖系数','高血糖年龄','高血糖甘油三酯','高血糖*r-谷氨酰基转换酶',
       '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
       '脂肪肝系数','血脂系数','甘油三酯排名','*r-谷氨酰基转换酶排名','年龄排名','*丙氨酸氨基转换酶排名','四指标综合排名',
       '血糖']]

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
test.ix[test['年龄']<=63,'高血糖年龄'] = 0
test.ix[test['高血糖年龄']!=0,'高血糖年龄'] = 1
test.ix[test['甘油三酯']<=2.80,'高血糖甘油三酯'] = 0
test.ix[test['高血糖甘油三酯']!=0,'高血糖甘油三酯'] = 1
test.ix[test['*r-谷氨酰基转换酶']<=58.585,'高血糖*r-谷氨酰基转换酶'] = 0
test.ix[test['高血糖*r-谷氨酰基转换酶']!=0,'高血糖*r-谷氨酰基转换酶'] = 1
test.ix[test['*丙氨酸氨基转换酶']<=38.495,'高血糖*丙氨酸氨基转换酶'] = 0
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


test = test[['是否空值','危险系数','体检日期','id','性别','年龄',
          '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶','高血糖系数','高血糖年龄','高血糖甘油三酯','高血糖*r-谷氨酰基转换酶',
       '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
       '脂肪肝系数','血脂系数','甘油三酯排名','*r-谷氨酰基转换酶排名','年龄排名','*丙氨酸氨基转换酶排名','四指标综合排名'
       ]]

def make_feat(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])
    data['性别'] = data['性别'].map({'男':1,'女':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
    data.fillna(data.median(axis=0),inplace=True)
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    return train_feat,test_feat


train_feat,test_feat = make_feat(train,test)
predictors = [f for f in test_feat.columns if f not in ['血糖']]


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5mse',score,False)

print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

print('开始CV 5折训练...')
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=10)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'],categorical_feature=['年龄'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['血糖'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:,i] = gbm.predict(test_feat[predictors])
print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'],train_preds)*0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
submission.to_csv('D:/WinCode/自如数据/血糖预测.csv',header=None,index=False, float_format='%.4f')