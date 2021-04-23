# -*- encoding:utf-8 -*-
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
'''
1 简单介绍任务
2 介绍数据来源,数据介绍
3 三种方式揭示特征准确性
    3.1 皮尔森相关系数
    3.2 特征重要性
    3.3 n-1个特征,n种情况找出波动较大的,少的特征为重要特征
'''

# 加载数据
titanic = pd.read_csv('text.csv', encoding="gbk")
# 获取特征列
attribute_all = ['region', 'day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'Rain', 'FFMC',
                 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
# Classes 决策类:即'火灾'和'非火灾'
y = titanic['Classes']


'''
    3.1 皮尔森相关系数
    按行计算皮尔逊相关系数,计算各列之间的相关系数,cm输出为相关系数矩阵
'''
cols = attribute_all + ['Classes']
cm = np.corrcoef(titanic[cols][::2].values.T)
# 画图相关系数矩阵,font_scale设置字体大小
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols
                 )
plt.show()

'''
    3.2 特征重要性
'''
X = titanic[attribute_all]
# 将数据进行分割为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# 保证不同特征组合训练集测试集数据一致性
X_train, X_test, y_train, y_test = X[::2], X[1::2], y[::2], y[1::2]
# 初始化决策树,基尼
clf = DecisionTreeClassifier(criterion="gini")
# 训练
clf = clf.fit(X_train, y_train)
# 验证集查看得分
score = clf.score(X_test, y_test)

print('score得分:', score)
print('特征重要性 :', list(clf.feature_importances_))

plt.bar(attribute_all, list(clf.feature_importances_))
plt.xlabel('attribute')
plt.ylabel('score')
plt.show()

'''
    3.3 n-1个特征,n种情况找出波动较大的,少的特征为重要特征
'''
# score_all:模型得分,no_attribute_all:减少的特征
score_all = []
no_attribute_all = []

# n-1个特征,遍历创建决策树,获取得分准确性
for attribute in list(combinations(attribute_all, len(attribute_all)-1)):

    X = titanic[list(attribute)]
    # 将数据进行分割为训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # 保证不同特征组合训练集测试集数据一致性
    X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test
    # 初始化决策树,基尼
    clf = DecisionTreeClassifier(criterion="gini")
    # 加载训练
    clf = clf.fit(X_train, y_train)
    # 验证集查看得分
    score = clf.score(X_test, y_test)

    print('score得分:', score)
    # 记录得分,缺少的特征
    score_all.append(score)
    no_attribute_all.append(list(set(attribute_all).difference(set(attribute)))[0])

# 分画图展示
plt.bar(no_attribute_all, score_all)
plt.xlabel('no_attribute')
plt.ylabel('score')
plt.show()
