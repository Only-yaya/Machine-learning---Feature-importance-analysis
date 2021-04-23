# Machine learning - Feature importance analysis
基于森林火灾数据集机器学习_特征的重要性分析

(1)皮尔森相关系数

(2)决策树模型feature_importances方法

(3)基于决策树模型数据操作
利用数据差额筛选的方法，例如有n个特征，但是只使用其中n-1个特征构建决策树模型，那么便有n中选择，分别拿n个n-1的数据集做训练，比对不同的决策树模型识别效果来判断哪个特征最重要
