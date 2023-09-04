import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import VotingClassifier
import joblib


# 标签编号
def labelEncode(df, s):
    dic = {}
    uni = df.unique()
    for i in range(len(uni)):
        dic[uni[i]] = i
    print("{}标签转换为：".format(s), dic, '\n')
    with open('results/labels/{}.dic'.format(s), 'w+') as f:
        f.write(str(dic))
        f.close()
    return df.replace(dic), dic

# 把不是数字的列都转为编号
def colsEncode(df):
    for i in df.columns:
        if df[i].dtype == 'object':
            df[i], dic = labelEncode(df[i], i)
    return df

# 保存模型
def saveModels(clf):
    joblib.dump(clf, 'results/models/{}.pkl'.format(label))
    print(label, '保存成功')


# 计算准确率
def calcAR(clf):
    global x_train, x_test, y_train, y_test

    y_pre = clf.predict(x_test)
    print('{}在预测集模型的准确率为：'.format(label), metrics.accuracy_score(y_test, y_pre))
    print('{}在训练集模型的准确率为：'.format(label), metrics.accuracy_score(y_train, clf.predict(x_train)))
    tem = metrics.accuracy_score(y, clf.predict(x))
    print('{}的综合准确率为：'.format(label), tem)

    t = pd.DataFrame(y_test - y_pre)
    print('{}标签准确率：'.format(label), len(t[t[0] == 0]) / len(t[0]))
    t = pd.DataFrame(y_pre)
    t[t[0] != 0] = 1
    yt = pd.DataFrame(y_test)
    yt[yt[0] != 0] = 1
    print('{}拦截准确率：'.format(label), len(yt[((yt == t)[0] == True)]) / len(yt))

    return tem


if __name__ == '__main__':

    #################################数据预处理#################################
    train_data = pd.read_csv(r'UNSW_NB15_training-set.csv')
    # 处理标签
    y, _ = labelEncode(train_data['attack_cat'], 'attack_cat')
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
    #################################数据预处理#################################


    #################################模型训练评估#################################
    # # 模型实例化
    # knn = KNeighborsClassifier()
    # LR = LogisticRegression()
    # Ada = ada()
    # GBDT = GradientBoostingClassifier()
    # svc = SVC(probability=True)
    # rf = RF()
    #
    # weight = []
    #
    # # 集合模型训练并评估
    # for clf, label in zip([knn, LR, Ada, GBDT, svc, rf],
    #                       ['knn', 'LR', 'Ada', 'GBDT', 'svc', 'rf']):
    #     clf.fit(x_train, y_train)
    #     weight.append(calcAR(clf))
    #     saveModels(clf)
    #
    # label = 'vote'
    # w = weight/sum(weight)
    # vote = VotingClassifier(estimators=[('knn', knn), ('LR', LR), ('Ada', Ada),
    #                                     ('GBDT', GBDT), ('svc', svc), ('rf', rf)],
    #                         voting='soft', weights=weight)
    # vote.fit(x_train, y_train)
    # saveModels(vote)
    # calcAR(vote)

    # 读取模型并评估
    for label in zip(['knn', 'LR', 'Ada', 'GBDT', 'svc', 'rf', 'vote']):
        clf = joblib.load('results/models/{}.pkl'.format(label))
        clf.fit(x_train, y_train)
        calcAR(clf)
    #################################模型训练评估#################################