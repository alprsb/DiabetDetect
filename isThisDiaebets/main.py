import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

dataFrame = pd.read_csv("diabetes.csv")
#print(dataFrame.head())

seker_hastalari = dataFrame[dataFrame.Outcome == 1]
saglikli_insanlar = dataFrame[dataFrame.Outcome == 0]

plt.scatter(saglikli_insanlar.Age,saglikli_insanlar.Glucose,color="green",label="Sağlıklı İnsanlar",alpha=0.4)
plt.scatter(seker_hastalari.Age,seker_hastalari.Glucose,color="red",label="Hastalıklı İnsanlar",alpha=0.4)
plt.xlabel= "Age"
plt.ylabel = "Glucose"
plt.legend()
#plt.show()

y = dataFrame["Outcome"].values
x_raw = dataFrame.drop("Outcome",axis=1).values

scaler = MinMaxScaler()
x = scaler.fit_transform(x_raw)

x_raw = pd.DataFrame(x_raw)
x = pd.DataFrame(x)

#print("Normalizasyondan önce:\n")
#print(x_raw.head())
#print("\nNormalizasyondan sonra:\n")
#print(x.head())


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=1)

knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("K=3 için Test verilerimizin doğrulama testi sonucu :%",knn.score(x_test, y_test) * 100)


for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors = k)
    knn_yeni.fit(x_train,y_train)
    print(k, "  ", "Doğruluk oranı: %", knn_yeni.score(x_test,y_test)*100)


new_prediction = knn.predict(scaler.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
print(new_prediction[0])