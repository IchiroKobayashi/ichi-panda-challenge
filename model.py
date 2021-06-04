from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

"""
▽学習手法
・ロジスティック回帰
・サポートベクターマシン
・k-最近傍法
・決定木
・ランダムフォレスト
・ニューラルネットワーク
"""

def get_models()->list:
    models = []
    models.append(("LogisticRegression", LogisticRegression(max_iter=5000)))
    models.append(("SVC", SVC()))
    # models.append(("LinearSVC", LinearSVC(dual=False)))
    # models.append(("KNeighbors", KNeighborsClassifier()))
    # models.append(("DecisionTree", DecisionTreeClassifier()))
    # models.append(("RandomForest", RandomForestClassifier()))
    # models.append(("MLPClassifier", MLPClassifier(solver='lbfgs', random_state=0, max_iter=5000)))
    return models

if __name__ == "__main__":
    get_models()