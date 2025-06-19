import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, Callback
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFrame, QSizePolicy
)
from tensorflow.keras.callbacks import EarlyStopping, Callback
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

# Импорт окон
from perceptron import PerceptronDataSelection


class NeuralNetSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Конструктор нейросетей")
        self.setFixedSize(700, 500)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Заголовок
        title = QLabel("Выберите тип нейросети")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Карточки в 2 строки по 2
        grid_layout = QVBoxLayout()
        top_row = QHBoxLayout()
        bottom_row = QHBoxLayout()

        top_row.addWidget(self.create_card(
            "Перцептрон",
            "Простой многослойный перцептрон. Для регрессии и базовой классификации.",
            "icons/perceptron.png",
            self.open_perceptron_window
        ))
        top_row.addWidget(self.create_card(
            "Сверточная сеть",
            "Классификация изображений, распознавание объектов.",
            "icons/cnn.png",
            self.open_cnn_window
        ))

        bottom_row.addWidget(self.create_card(
            "Рекуррентная сеть",
            "Анализ звуков и текстов, работа с последовательностями.",
            "icons/rnn.png",
            self.open_rnn_window
        ))

        grid_layout.addLayout(top_row)
        grid_layout.addSpacing(10)
        grid_layout.addLayout(bottom_row)
        main_layout.addLayout(grid_layout)



        self.setLayout(main_layout)



    def create_card(self, title_text, description, icon_path, on_click=None):
        card = QFrame()
        card.setFixedSize(300, 120)
        card.setStyleSheet("""
            QFrame {
                background-color: #F5F5F5;
                border: 1px solid #DDDDDD;
                border-radius: 12px;
            }
            QFrame:hover {
                background-color: #EDEDED;
            }
        """)

        layout = QHBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Иконка
        icon_label = QLabel()
        pixmap = QPixmap(icon_path).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("background: transparent; border: none;")
        icon_label.setAttribute(Qt.WA_TransparentForMouseEvents)

        # Текст
        text_layout = QVBoxLayout()
        text_layout.setSpacing(5)

        title_label = QLabel(title_text)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setStyleSheet("background: transparent; border: none; color: #333333;")
        title_label.setAttribute(Qt.WA_TransparentForMouseEvents)

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("background: transparent; border: none; color: #666666; font-size: 11px;")
        desc_label.setAttribute(Qt.WA_TransparentForMouseEvents)

        text_layout.addWidget(title_label)
        text_layout.addWidget(desc_label)

        layout.addWidget(icon_label)
        layout.addLayout(text_layout)

        card.setLayout(layout)

        if on_click:
            card.mousePressEvent = lambda event: on_click()

        return card

    # Методы открытия окон
    def open_perceptron_window(self):
        self.perceptron_window = PerceptronDataSelection()
        self.perceptron_window.show()

    def open_cnn_window(self):
        from cnn import CNNClassifierWindow
        self.cnn_window = CNNClassifierWindow()
        self.cnn_window.show()

    def open_rnn_window(self):
        from rnn import RNNClassifierWindow
        self.rnn_window = RNNClassifierWindow()
        self.rnn_window.show()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuralNetSelector()
    window.show()
    sys.exit(app.exec_())
