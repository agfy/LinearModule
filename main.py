import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('perceptron-train.csv', header=None)
test_data = pd.read_csv('perceptron-test.csv', header=None)

X_train = train_data.as_matrix(columns=[train_data.columns[1], train_data.columns[2]])
y_train = train_data.as_matrix(columns=[train_data.columns[0]])
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

X_test = test_data.as_matrix(columns=[test_data.columns[1], test_data.columns[2]])
y_test = test_data.as_matrix(columns=[test_data.columns[0]])

predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
clf1 = Perceptron(random_state=241)
clf1.fit(X_train_scaled, y_train)

X_test_scaled = scaler.transform(X_test)
predictions_scaled = clf1.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, predictions_scaled)
diff = accuracy_scaled - accuracy