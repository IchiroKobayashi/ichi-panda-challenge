import numpy as np
import pandas as pd
import copy
import model
from sklearn.model_selection import cross_val_score
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
    train_corrected = correct_data(copy.deepcopy(train_raw)) # 参照値、train_rawも変わる
    test_corrected  = correct_data(copy.deepcopy(test_raw))
    # print(train_raw.head(10)) 
    # print(train_corrected.head(10)) 
    train_corr = train_corrected.corr()
    # print(train_corr)
    # 使用するカラムを指定：チケットのクラス、性別、年齢、タイタニック号に乗っていた兄弟と配偶者の数、タイタニック号に乗っていた両親と子どもの数、旅客運賃、乗船場
    results = []
    names = []
    models = model.get_models()
    for name, m in models:
        # 各モデルを使って交差検証を実施、x:使用するデータセット(カラム指定)、y:予測する値
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



def correct_data(titanic_data):
    titanic_data.Age = titanic_data.Age.fillna(titanic_data.Age.median())
    titanic_data.Sex = titanic_data.Sex.replace(['male', 'female'], [0, 1])
    titanic_data.Embarked = titanic_data.Embarked.fillna("S") # これはバイアス。主観入りすぎ
    titanic_data.Embarked = titanic_data.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
    titanic_data.Fare = titanic_data.Fare.fillna(titanic_data.Fare.median())
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
    algo.fit(train_corrected[predictors], train_corrected["Survived"])
    predictions = algo.predict(test_corrected[predictors])
    submission = pd.DataFrame({
            "PassengerId": test_corrected["PassengerId"],
            "Survived": predictions
        })
    submission.to_csv('./result/submission.csv', index=False)


if __name__ == "__main__":
    exec()