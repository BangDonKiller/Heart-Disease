import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('./dataset/heart.csv')
X = data.drop('target', axis=1).values
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)

rf.fit(X_train, y_train)

predictions = rf.predict(X_test)

print("Predictions:", predictions)

print("True:", y_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
