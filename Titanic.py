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
