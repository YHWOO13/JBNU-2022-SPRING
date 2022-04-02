from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from pandas import DataFrame

# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
digit = datasets.load_digits()
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size = 0.6)

SVM_set = [[0.01,'linear'], [1,'linear'], [100,'linear'],
           [0.01,'poly'], [1,'poly'], [100,'poly'],
           [0.01,'rbf'], [1,'rbf'], [100,'rbf'],
           [0.01,'sigmoid'], [1,'sigmoid'], [100,'sigmoid']]
SVM_accuracy = []

RF_set = [[10, 10], [50, 10], [100, 10],
          [10, 50], [50, 50], [100, 50],
          [10, 100], [50, 100], [100, 100]]
RF_accuracy = []

def s_v_m(c,k):
    s = svm.SVC(C = c, kernel = k)
    s.fit(x_train, y_train)

    # print(i[0],i[1])
    res = s.predict(x_test)

    return res

def s_v_m_test():
    step = 0
    for i in SVM_set:
        # 혼동 행렬 구함
        conf = np.zeros((10, 10))
        each_res = s_v_m(i[0], i[1])
        for j in range(len(each_res)):
            conf[each_res[j]][y_test[j]] += 1

        # 정확률 측정하고 출력
        no_correct = 0
        for k in range(10):
            no_correct += conf[k][k]
        accuracy = no_correct/len(each_res)
        accuracy = round(accuracy * 100, 4)

        SVM_accuracy.append(accuracy)

        sent = f'C_parameter:{i[0]}이며, {i[1]}분류를 이용한 테스트 집합 정확률은{accuracy} %입니다.'
        # print(sent)

    svm_data = {'linear': SVM_accuracy[0:3],
               'poly': SVM_accuracy[3:6],
               'rbf': SVM_accuracy[6:9],
                'sigmoid': SVM_accuracy[9:12]}

    SVM_df = DataFrame(svm_data)
    SVM_df = SVM_df.rename(index={0:'0.01', 1:'1', 2:'100'})

    print(SVM_df)

def randomforest(e, md):
    rf = RandomForestClassifier(n_estimators = e, max_depth = md)
    rf.fit(x_train, y_train)
    pre = rf.predict(x_test)

    return pre

def rf_test():
    for i in RF_set:

        conf = np.zeros((10, 10))
        # each_res = s_v_m(i[0], i[1])
        pre = randomforest(i[0], i [1])
        for j in range(len(pre)):
            conf[pre[j]][y_test[j]] += 1

        # 정확률 측정하고 출력
        no_correct = 0
        for k in range(10):
            no_correct += conf[k][k]
        accuracy = no_correct/len(pre)
        accuracy = round(accuracy * 100, 4)

        RF_accuracy.append(accuracy)

    RF_data = {'max_Depth 10': RF_accuracy[0:3],
               '50': RF_accuracy[3:6],
               '100': RF_accuracy[6:9],}

    RF_df = DataFrame(RF_data)
    RF_df = RF_df.rename(index={0: 'n_estimator 10', 1: '50', 2: '100'})

    print(RF_df)
        # # print(accuracy)
        # sent = f'n_estimators:{i[0]}이며, max_depth: {i[1]}. 테스트 집합 정확률은{accuracy} %입니다.'
        #
        # print(sent)

s_v_m_test()
rf_test()
