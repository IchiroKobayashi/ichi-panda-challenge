import numpy as np
import pandas as pd
import copy
import model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
"""
▽Data Set
・PassengerId：データにシーケンシャルでついている番号
・Survived：生存（0 = No, 1 = Yes）　訓練用データにのみ存在
・Pclass：チケットのクラス（1 = 1st, 2 = 2nd, 3 = 3rd）
・Name：名前
・Sex：性別
・Age：年齢
・SibSp：タイタニック号に乗っていた兄弟と配偶者の数
・Parch：タイタニック号に乗っていた両親と子どもの数
・Ticket：チケット番号
・Fare：旅客運賃
・Cabin：船室番号
・Embarked：乗船場（C = Cherbourg, Q = Queenstown, S = Southampton）
"""
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def exec():
    train_raw = pd.read_csv("./seed/titanic_train.csv", dtype={"Age": np.float64}, )
    test_raw  = pd.read_csv("./seed/titanic_test.csv", dtype={"Age": np.float64}, )
    train_corrected, test_corrected = correct_data(copy.deepcopy(train_raw), copy.deepcopy(test_raw)) # 参照値、train_rawも変わる
    # print(train_raw.head(10)) 
    # train_corr = train_corrected.corr()
    # print(train_corr)
    # 使用するカラムを指定：チケットのクラス、性別、年齢、タイタニック号に乗っていた兄弟と配偶者の数、タイタニック号に乗っていた両親と子どもの数、旅客運賃、乗船場
    results = []
    names = []
    models = model.get_models()
    for name, m in models:
        # 各モデルを使って交差検証を実施、x:使用するデータセット(カラム指定)、y:予測する値、cv:○個あるデータブロックのうち、1個を検証用データ、その他を訓練データとして使う
        result = cross_val_score(m, train_corrected[predictors], train_corrected["Survived"], cv=3)
        names.append(name)
        results.append(result)

    best_model = get_best_model(names, results)
    print(best_model)
    algo = None
    if best_model[0]['name'] == 'LogisticRegression':
        algo = model.LogisticRegression(max_iter=5000)
    elif best_model[0]['name'] == 'SVC':
        algo = model.SVC()
    elif best_model[0]['name'] == 'LinearSVC':
        algo = model.LinearSVC(dual=False)
    elif best_model[0]['name'] == 'KNeighbors':
        algo = model.KNeighborsClassifier()
    elif best_model[0]['name'] == 'DecisionTree':
        algo = model.DecisionTreeClassifier()
    elif best_model[0]['name'] == 'RandomForest':
        algo = model.RandomForestClassifier()
    else:
        algo = model.MLPClassifier(solver='lbfgs', random_state=0, max_iter=5000)

    output_result_as_csv(algo, train_corrected, test_corrected)

def correct_data(train_raw, test_raw):
    train_raw.Age = train_raw.Age.fillna(test_raw.Age.median()) # テストデータのmedianをとったほうが精度向上する。
    train_raw.Fare = train_raw.Fare.fillna(test_raw.Fare.median()) # テストデータのmedianをとったほうが精度向上する。
    test_raw.Age = test_raw.Age.fillna(test_raw.Age.median())
    test_raw.Fare = test_raw.Fare.fillna(test_raw.Fare.median())
    train_data = correct_common_data(train_raw)
    test_data = correct_common_data(test_raw)
    return train_data, test_data

def correct_common_data(titanic_data):
    titanic_data.Sex = titanic_data.Sex.replace(['male', 'female'], [0, 1])
    titanic_data.Embarked = titanic_data.Embarked.fillna("S") # これはバイアス。主観入りすぎ
    titanic_data.Embarked = titanic_data.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
    return titanic_data

def get_best_model(names, results):
    best_model = []
    for i in range(len(names)):
        if i == 0:
            # 最初は比較対象の元となるので入れる
            best_model.append({'name':names[i], 'result':results[i].mean()}.copy())
        elif best_model[0]['result'] < results[i].mean():
            # 比較して、数値が良い方をセット
            best_model[0] = {'name':names[i], 'result':results[i].mean()}
    return best_model

def output_result_as_csv(algo, train_corrected, test_corrected):
    # param_grid = {
    #     "max_depth": [3, None], #distribution
    #     "n_estimators":[50,100,200,300,400,500],
    #     "max_features": sp_randint(1, 11),
    #     "min_samples_split": sp_randint(2, 11),
    #     "min_samples_leaf": sp_randint(1, 11),
    #     "bootstrap": [True, False],
    #     "criterion": ["gini", "entropy"]
    # }
    # グリッドサーチによるハイパーパラメータの最適化
    # gsc = GridSearchCV(algo, parameters, cv=3)
    # rsc = RandomizedSearchCV(estimator=algo, param_distributions=param_grid, cv=5, n_iter=30) # scoring="accuracy"
    algo.fit(train_corrected[predictors], train_corrected["Survived"])
    predictions = algo.predict(test_corrected[predictors])
    submission = pd.DataFrame({
            "PassengerId": test_corrected["PassengerId"],
            "Survived": predictions
        })
    submission.to_csv('./result/submission.csv', index=False)


if __name__ == "__main__":
    exec()