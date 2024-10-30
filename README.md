Titanic Survival Prediction

此專案是針對Kaggle上的Titanic競賽而設計，目標是預測在泰坦尼克號沉船事件中乘客的生存情況。此專案使用Python和隨機森林模型來進行預測分析。

專案文件結構

train.csv：訓練數據集，包含已知生存情況的乘客資料。
test.csv：測試數據集，用於預測乘客生存情況。
submission.csv：生成的預測結果檔案，符合Kaggle的提交格式。
前置需求

本專案依賴以下Python套件：

pandas：用於資料讀取與處理
seaborn：用於數據視覺化
matplotlib：用於數據視覺化
scikit-learn：用於機器學習模型和數據分割
安裝這些套件：

bash
複製程式碼
pip install pandas seaborn matplotlib scikit-learn
使用步驟

1. 資料準備
下載train.csv和test.csv，並將這些文件放入專案的資料夾中。您可以在Kaggle Titanic競賽頁面下載資料集。

2. 執行程式碼
將以下程式碼儲存為titanic_prediction.py，並執行以產生預測結果。

python
複製程式碼
# 匯入所需的套件
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 讀取訓練和測試資料
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 資料清理與特徵工程
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# 將文字類別變數轉換為數字
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 選擇與生存最相關的特徵
X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = train_data['Survived']
X_test = test_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

# 訓練模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 進行預測
predictions = model.predict(X_test)

# 將預測結果寫入 CSV 文件
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Prediction file created: submission.csv")
執行以下命令來生成submission.csv預測文件：

bash
複製程式碼
python titanic_prediction.py
3. 提交結果至Kaggle
程式碼執行完成後會生成submission.csv文件。此檔案包含兩個欄位：

PassengerId：乘客ID
Survived：預測的生還標籤（1表示生還，0表示未生還）
上傳此文件至Kaggle競賽頁面的「Submit Predictions」區域以進行評分。

程式碼說明

資料清理與特徵工程
處理缺失值並將文字變數轉換為數字以便於模型處理：

填補Age欄位缺失值為中位數。
填補Embarked欄位缺失值為眾數。
將Sex和Embarked轉換為數字編碼。
模型訓練
使用隨機森林分類器對訓練集進行訓練，並預測測試集生存情況。

python
複製程式碼
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
評估模型
可以使用train_test_split方法分割訓練集以進行驗證。以下是驗證的準確度計算範例：

python
複製程式碼
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy}')
結果提交與評估

您可以在Kaggle的競賽頁面上傳submission.csv以評估模型的表現，並查看在Leaderboard上的排名。

參考資料

Kaggle Titanic Competition
Scikit-learn RandomForest Documentation
