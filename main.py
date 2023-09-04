import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier


# 标签编号
def labelEncode(df):
    dic = {}
    uni = df.unique()
    for i in range(len(uni)):
        dic[uni[i]] = i
    print("标签转换为：", dic, '\n')
    return df.replace(dic), dic


# 把不是数字的列都转为编号
def colsEncode(df):
    for i in df.columns:
        if df[i].dtype == 'object':
            df[i], dic = labelEncode(df[i])
    return df


# 计算准确率
def calcAR(clf):
    global x_train, x_test, y_train, y_test

    y_pre = clf.predict(x_test)
    print('{}在预测集模型的准确率为：'.format(label), metrics.accuracy_score(y_test, y_pre))
    print('{}在训练集模型的准确率为：'.format(label), metrics.accuracy_score(y_train, clf.predict(x_train)))
    print('{}的综合准确率为：'.format(label), metrics.accuracy_score(y, clf.predict(X)))

    t = pd.DataFrame(y_test - y_pre)
    print('{}标签准确率：'.format(label), len(t[t[0] == 0]) / len(t[0]))
    t = pd.DataFrame(y_pre)
    t[t[0] != 0] = 1
    yt = pd.DataFrame(y_test)
    yt[yt[0] != 0] = 1
    print('{}拦截准确率：'.format(label), len(yt[((yt == t)[0] == True)]) / len(yt))


if __name__ == '__main__':
    train_data = pd.read_csv(r'UNSW_NB15_training-set.csv')

    # 处理标签
    y, ydic = labelEncode(train_data['attack_cat'])
    y = np.array(y)
    # 处理输入数据
    X = colsEncode(train_data.drop(['attack_cat', 'label'], axis=1))
    X = np.array(X)
    # 分割训练数据和测试数据
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # 数据正规化
    tranfer = StandardScaler()
    x = tranfer.fit_transform(X)
    x_train = tranfer.transform(x_train)
    x_test = tranfer.transform(x_test)

    # 模型实例化
    knn = KNeighborsClassifier()
    LR = LogisticRegression()
    Ada = ada()
    GBDT = GradientBoostingClassifier()
    svc = SVC()
    rf = RF()

    # 集合模型训练
    for clf, label in zip([knn, LR, Ada, GBDT, svc, rf],
                          ['KNN', 'LR', 'Ada', 'GBDT', 'svc', 'rf']):
        clf.fit(x_train, y_train)
        calcAR(clf)
