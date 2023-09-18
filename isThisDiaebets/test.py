import sys

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from PyQt5.QtGui import QFont, QPixmap, QColor, QPalette, QPainter, QRegion
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve

dataFrame = pd.read_csv("diabetes.csv")

y = dataFrame["Outcome"].values
x_raw = dataFrame.drop("Outcome", axis=1).values

scaler = MinMaxScaler()
x = scaler.fit_transform(x_raw)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

class AnimatedLogoLabel(QLabel):
    def __init__(self, logo_path, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)
        self.setPixmap(QPixmap(logo_path).scaled(40, 40))
        self.animation = QPropertyAnimation(self, b'size', self)
        self.animation.setDuration(1700)
        self.animation.setEasingCurve(QEasingCurve.OutElastic)

    def animate(self, size):
        self.animation.setStartValue(self.size())
        self.animation.setEndValue(size)
        self.animation.start()

class DiabetesPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Diabetes Predictor')
        self.setFixedSize(350, 450)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        self.setPalette(palette)

        self.labels = []
        self.inputs = []

        self.layout = QVBoxLayout()

        feature_logo_paths = [("Pregnancies", "pregnancy.png"),
                              ("Glucose", "glucose.png"),
                              ("BloodPressure", "bloodpressure.png"),
                              ("SkinThickness", "skinthickness.png"),
                              ("Insulin", "insulin.png"),
                              ("BMI", "bmi.png"),
                              ("DiabetesPedigreeFunction", "pedigree.png"),
                              ("Age", "age.png")]

        for feature, logo_path in feature_logo_paths:
            label_layout = QHBoxLayout()
            label = AnimatedLogoLabel(logo_path, self)
            label.setStyleSheet("background-color: transparent;")
            label.setFont(QFont("Arial", 12))
            input_box = QLineEdit(self)
            input_box.setFont(QFont("Arial", 12))
            input_box.setStyleSheet("background-color: white;")
            input_box.setPlaceholderText(feature)

            label_layout.addWidget(label)
            label_layout.addWidget(input_box)

            self.labels.append(label)
            self.inputs.append(input_box)

            self.layout.addLayout(label_layout)

        self.predict_button = QPushButton('Tahmin Et', self)
        self.predict_button.setFont(QFont("Arial", 14))
        self.predict_button.setStyleSheet("background-color: #007acc; color: white;")
        self.predict_button.clicked.connect(self.predict_diabetes)
        self.layout.addWidget(self.predict_button)

        self.result_label = QLabel(self)
        self.result_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.result_label.setStyleSheet("color: #007acc;")
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

        for i, input_box in enumerate(self.inputs):
            input_box.textChanged.connect(lambda text, idx=i: self.animate_logo(idx))

    def animate_logo(self, idx):
        self.labels[idx].animate(QSize(45, 45))

    def predict_diabetes(self):
        input_data = [input_box.text() for input_box in self.inputs]


        if not all(input_data):
            print("Lütfen tüm özellikleri girin.")
            return

        input_data = [float(value) if value else 0.0 for value in input_data]  # Boş girişleri 0.0 olarak kabul et
        input_data = scaler.transform([input_data])
        new_prediction = knn.predict(input_data)

        if new_prediction[0] == 1:
            self.result_label.setText('Sonuç: Şeker Hastası')
        else:
            self.result_label.setText('Sonuç: Sağlıklı')

def main():
    app = QApplication(sys.argv)
    window = DiabetesPredictor()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()