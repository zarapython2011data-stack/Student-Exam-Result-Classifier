import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('Main.csv')

X=df[['study_hours', 'attendance']]
y=df['passed']

X_train ,X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)

model=LogisticRegression()
model.fit(X_train , y_train)

y_pred=model.predict(X_test)

accuracy= accuracy_score(y_test, y_pred)
print("Model Accuracy:" , accuracy)