import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

diabetes = pd.read_csv("diabetes.csv")

pd.options.display.max_columns = None
print(diabetes.head(15))

print(diabetes.info())
print(diabetes.describe())

sns.barplot(data=diabetes, x='Age', y='BloodPressure')
plt.show()

dist = sns.displot(data=diabetes, x='Age', color='red')
#dist.set_xlabels('Starost')
plt.xlabel('Starost')
plt.show()

diabetes_filtered = diabetes[diabetes['Outcome'] == 1]
diabetes_filtered = diabetes_filtered[diabetes_filtered['Age'] > 40]
print(diabetes_filtered.head())

diabetes_filtered['DiabetesPedigreeFunction'] = diabetes_filtered['DiabetesPedigreeFunction'].mean()
diabetes_filtered['Glucose'] = diabetes_filtered['BMI'].min()
print(diabetes_filtered['Glucose'].head(15))
print(diabetes_filtered['DiabetesPedigreeFunction'].head(15))

# print(diabetes_filtered['DiabetesPedigreeFunction'].mean())
# print(diabetes_filtered['BMI'].min())

# sns.heatmap(data=diabetes)
# plt.show()

cor = diabetes.corr()
sns.heatmap(cor,annot=True)
plt.show()

diabetes.drop(columns=['BloodPressure', 'SkinThickness'], inplace=True)
#print(diabetes.head())

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

y=diabetes['Outcome']
X=diabetes.drop('Outcome', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predict = model.predict(X_test)
print('Stablo odlucivanja ima numericku vrednost tacnosti: ', accuracy_score(predict, y_test))