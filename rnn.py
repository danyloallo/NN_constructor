import sys
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QListWidget, QListWidgetItem, QCheckBox,
                             QSpinBox, QLabel, QGroupBox, QFormLayout, QProgressBar, QComboBox, QTableWidget,
                             QTableWidgetItem)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns


class RNNClassifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Реккурентая нейросеть для классификации звуков")
        self.setGeometry(100, 100, 800, 600)
        self.dataset_path = ""
        self.classes = []
        self.selected_classes = []
        self.model = None
        self.history = None
        self.n_mfcc = 13
        self.hop_length = 512
        self.sr = 16000
        self.units_spins = []  # Список для хранения полей ввода нейронов
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Dataset selection
        dataset_group = QGroupBox("Выбор датасета")
        dataset_layout = QVBoxLayout()

        self.select_dataset_btn = QPushButton("Выберите папку датасета")
        self.select_dataset_btn.setToolTip(
            "Выберите папку с подпапками, где каждая подпапка соответствует классу аудиофайлов.\nРекомендация: Убедитесь, что файлы в формате WAV.")
        self.select_dataset_btn.clicked.connect(self.select_dataset)
        dataset_layout.addWidget(self.select_dataset_btn)

        self.class_list = QListWidget()
        self.class_list.setEnabled(False)
        self.class_list.setToolTip(
            "Выберите классы для классификации.\nРекомендация: Выберите 2–10 классов с достаточным количеством аудиофайлов.")
        dataset_layout.addWidget(QLabel("Выберите классы:", toolTip="Отметьте классы для обучения модели."))
        dataset_layout.addWidget(self.class_list)

        dataset_group.setLayout(dataset_layout)
        controls_layout.addWidget(dataset_group)

        # Model configuration
        model_group = QGroupBox("Настройка модели")
        self.model_layout = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Ручная настройка", "Автоматическая настройка"])
        self.mode_combo.setToolTip(
            "Выберите способ настройки модели.\nРучная настройка: задайте параметры RNN вручную.\nАвтоматическая настройка: использует предустановленные параметры.\nРекомендация: Используйте автоматическую настройку для быстрого старта.")
        self.model_layout.addRow("Тип настройки:", self.mode_combo)

        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 10)
        self.layers_spin.setValue(2)
        self.layers_spin.setToolTip("Количество слоёв LSTM в модели.\nРекомендация: 2–3 слоя для большинства задач.")
        self.layers_spin.valueChanged.connect(self.update_units_fields)
        self.model_layout.addRow("RNN слои:", self.layers_spin)

        # Создаём начальные поля для нейронов (по умолчанию 2 слоя)
        self.units_spins = []
        for i in range(self.layers_spin.value()):
            spin = QSpinBox()
            spin.setRange(32, 512)
            spin.setValue(128)
            spin.setToolTip(
                f"Количество нейронов в LSTM слое {i + 1}.\nРекомендация: 128–256 нейронов для аудиоданных.")
            self.units_spins.append(spin)
            self.model_layout.addRow(f"Нейронов в слое {i + 1}:", spin)

        self.dropout_spin = QSpinBox()
        self.dropout_spin.setRange(0, 50)
        self.dropout_spin.setValue(20)
        self.dropout_spin.setToolTip(
            "Процент нейронов, отключаемых в слоях для предотвращения переобучения.\nРекомендация: 20–30% для стабильного обучения.")
        self.model_layout.addRow("Dropout (%):", self.dropout_spin)

        model_group.setLayout(self.model_layout)
        controls_layout.addWidget(model_group)

        # Training controls
        train_group = QGroupBox("Обучение")
        train_layout = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        self.epochs_spin.setToolTip(
            "Количество эпох обучения.\nРекомендация: 10–20 для небольших датасетов, больше для сложных моделей.")
        train_layout.addRow("Эпохи:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 128)
        self.batch_spin.setValue(32)
        self.batch_spin.setToolTip(
            "Размер батча для обучения.\nРекомендация: 32 для большинства задач, уменьшайте при нехватке памяти.")
        train_layout.addRow("Размер батча:", self.batch_spin)

        self.train_btn = QPushButton("Обучить модель")
        self.train_btn.setToolTip("Запустить обучение модели на выбранном датасете.")
        self.train_btn.clicked.connect(self.train_model)
        train_layout.addRow(self.train_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setToolTip("Показывает прогресс обучения по эпохам.")
        train_layout.addRow(self.progress_bar)

        train_group.setLayout(train_layout)
        controls_layout.addWidget(train_group)

        # Model save/load
        io_group = QGroupBox("Сохранить или загрузить модель")
        io_layout = QHBoxLayout()

        self.save_btn = QPushButton("Сохранить модель")
        self.save_btn.setToolTip("Сохранить обученную модель в файл формата .h5.")
        self.save_btn.clicked.connect(self.save_model)
        io_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Загрузить модель")
        self.load_btn.setToolTip("Загрузить ранее сохранённую модель из файла .h5.")
        self.load_btn.clicked.connect(self.load_model)
        io_layout.addWidget(self.load_btn)

        io_group.setLayout(io_layout)
        controls_layout.addWidget(io_group)

        # Audio prediction
        predict_group = QGroupBox("Классификация")
        predict_layout = QHBoxLayout()

        self.predict_btn = QPushButton("Классифицировать звук")
        self.predict_btn.setToolTip("Классифицировать аудиофайл с помощью обученной модели.")
        self.predict_btn.clicked.connect(self.predict_audio)
        predict_layout.addWidget(self.predict_btn)

        self.predict_label = QLabel("Классифицировано: None")
        self.predict_label.setToolTip("Результат классификации аудиофайла (предсказанный класс).")
        predict_layout.addWidget(self.predict_label)

        predict_group.setLayout(predict_layout)
        controls_layout.addWidget(predict_group)

        main_layout.addWidget(controls_widget)

    def update_units_fields(self):
        """Обновляет поля ввода нейронов при изменении количества слоёв"""
        current_layers = self.layers_spin.value()
        current_spin_count = len(self.units_spins)

        # Удаляем старые поля, если их больше, чем нужно
        while len(self.units_spins) > current_layers:
            spin = self.units_spins.pop()
            # Удаляем строку из layout
            for i in range(self.model_layout.count()):
                layout_item = self.model_layout.itemAt(i)
                if layout_item.widget() == spin:
                    self.model_layout.removeRow(i)
                    break

        # Добавляем новые поля, если их меньше, чем нужно
        for i in range(current_spin_count, current_layers):
            spin = QSpinBox()
            spin.setRange(32, 512)
            spin.setValue(128)
            spin.setToolTip(
                f"Количество нейронов в LSTM слое {i + 1}.\nРекомендация: 128–256 нейронов для аудиоданных.")
            self.units_spins.append(spin)
            self.model_layout.insertRow(i + 2, f"Нейронов в слое {i + 1}:",
                                        spin)  # +2 для вставки после "Тип настройки" и "RNN слои"

    def select_dataset(self):
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Выберите папку датасета")
        if self.dataset_path:
            self.classes = [d for d in os.listdir(self.dataset_path)
                            if os.path.isdir(os.path.join(self.dataset_path, d))]
            self.class_list.clear()
            self.selected_classes = []
            for class_name in self.classes:
                item = QListWidgetItem()
                checkbox = QCheckBox(class_name)
                checkbox.setToolTip(f"Выберите класс {class_name} для включения в обучение.")
                checkbox.stateChanged.connect(self.update_selected_classes)
                self.class_list.addItem(item)
                self.class_list.setItemWidget(item, checkbox)
            self.class_list.setEnabled(True)

    def update_selected_classes(self):
        self.selected_classes = []
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            checkbox = self.class_list.itemWidget(item)
            if checkbox.isChecked():
                self.selected_classes.append(checkbox.text())

    def load_audio_data(self):
        if not self.selected_classes:
            self.predict_label.setText("Выберите хотя бы один класс!")
            return None, None, None, None

        X, y = [], []
        class_indices = {name: idx for idx, name in enumerate(self.selected_classes)}

        for class_name in self.selected_classes:
            class_path = os.path.join(self.dataset_path, class_name)
            audio_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            print(f"Класс {class_name}: {len(audio_files)} файлов")
            for audio_file in audio_files:
                file_path = os.path.join(class_path, audio_file)
                audio, sr = librosa.load(file_path, sr=self.sr)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
                X.append(mfcc.T)
                y.append(class_indices[class_name])

        if not X:
            self.predict_label.setText("Не найдены валидные аудиофайлы!")
            return None, None, None, None

        max_len = max(x.shape[0] for x in X)
        print(f"Максимальное количество временных шагов: {max_len}")
        X_padded = np.array([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant')
                             for x in X])
        y = np.array(y)

        X_train, X_val, y_train, y_val = train_test_split(X_padded, y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val

    def build_manual_model(self, input_shape, num_classes):
        model = models.Sequential()
        model.add(layers.Masking(mask_value=0., input_shape=input_shape))

        # Добавляем LSTM-слои с индивидуальным количеством нейронов
        for i in range(self.layers_spin.value()):
            units = self.units_spins[i].value()
            return_sequences = i < self.layers_spin.value() - 1  # Последний слой не возвращает последовательности
            model.add(layers.LSTM(units, return_sequences=return_sequences))
            model.add(layers.Dropout(self.dropout_spin.value() / 100))

        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

    def build_auto_model(self, input_shape, num_classes):
        model = models.Sequential([
            layers.Masking(mask_value=0., input_shape=input_shape),
            layers.LSTM(256, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(128),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def train_model(self):
        if not self.dataset_path or not self.selected_classes:
            self.predict_label.setText("Выберите датасет и хотя бы один класс!")
            return

        X_train, X_val, y_train, y_val = self.load_audio_data()
        if X_train is None or y_train is None:
            return

        num_classes = len(self.selected_classes)
        input_shape = (X_train.shape[1], X_train.shape[2])

        if self.mode_combo.currentText() == "Ручная настройка":
            self.model = self.build_manual_model(input_shape, num_classes)
        else:
            self.model = self.build_auto_model(input_shape, num_classes)

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.progress_bar.setRange(0, self.epochs_spin.value())
        try:
            self.history = self.model.fit(X_train, y_train,
                                          epochs=self.epochs_spin.value(),
                                          batch_size=self.batch_spin.value(),
                                          validation_data=(X_val, y_val),
                                          callbacks=[ProgressCallback(self.progress_bar)])
            self.plot_results()
            self.show_metrics_table()
            self.show_confusion_matrix(X_val, y_val)
        except Exception as e:
            self.predict_label.setText(f"Ошибка обучения: {str(e)}")
            return

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Потери')
        ax1.legend()

        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('Точность')
        ax2.legend()

        fig.tight_layout()
        plt.savefig('training_results.png')
        plt.show()

    def show_metrics_table(self):
        metric_window = QWidget()
        metric_window.setWindowTitle("Метрики обучения")
        metric_window.setGeometry(150, 150, 500, 300)
        layout = QVBoxLayout()

        table = QTableWidget()
        table.setRowCount(len(self.history.history['loss']))
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(
            ["Эпоха", "Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"])

        for i in range(len(self.history.history['loss'])):
            table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            table.setItem(i, 1, QTableWidgetItem(f"{self.history.history['loss'][i]:.4f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{self.history.history['val_loss'][i]:.4f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{self.history.history['accuracy'][i]:.4f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{self.history.history['val_accuracy'][i]:.4f}"))

        table.resizeColumnsToContents()
        layout.addWidget(table)
        metric_window.setLayout(layout)
        metric_window.show()
        self.metric_window = metric_window

    def show_confusion_matrix(self, X_val, y_val):
        y_pred = np.argmax(self.model.predict(X_val), axis=1)
        cm = confusion_matrix(y_val, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.selected_classes,
                    yticklabels=self.selected_classes)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.show()

    def save_model(self):
        if self.model:
            save_path = QFileDialog.getSaveFileName(self, "Сохранить модель", "", "H5 Files (*.h5)")[0]
            if save_path:
                try:
                    self.model.save(save_path)
                    self.predict_label.setText("Модель успешно сохранена!")
                except Exception as e:
                    self.predict_label.setText(f"Ошибка сохранения: {str(e)}")

    def load_model(self):
        load_path = QFileDialog.getOpenFileName(self, "Загрузить модель", "", "H5 Files (*.h5)")[0]
        if load_path:
            try:
                self.model = tf.keras.models.load_model(load_path)
                self.predict_label.setText("Модель успешно загружена!")
            except Exception as e:
                self.predict_label.setText(f"Ошибка загрузки: {str(e)}")

    def predict_audio(self):
        if not self.model or not self.selected_classes:
            self.predict_label.setText("Нет загруженной модели или не выбраны классы!")
            return

        audio_path = QFileDialog.getOpenFileName(self, "Выберите аудиофайл", "", "WAV Files (*.wav)")[0]
        if audio_path:
            audio, sr = librosa.load(audio_path, sr=self.sr)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
            mfcc = mfcc.T

            expected_time_steps = self.model.input_shape[1]
            current_time_steps = mfcc.shape[0]

            if current_time_steps > expected_time_steps:
                mfcc = mfcc[:expected_time_steps, :]
            else:
                mfcc = np.pad(mfcc, ((0, expected_time_steps - current_time_steps), (0, 0)),
                              mode='constant')

            mfcc = mfcc[np.newaxis, ...]
            print(f"MFCC shape for prediction: {mfcc.shape}")

            prediction = self.model.predict(mfcc)
            predicted_class = self.selected_classes[np.argmax(prediction)]
            confidence = np.max(prediction)
            self.predict_label.setText(f"Классифицировано: {predicted_class} (уверенность: {confidence:.2%})")


class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar):
        super().__init__()
        self.progress_bar = progress_bar

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.setValue(epoch + 1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RNNClassifierWindow()
    window.show()
    sys.exit(app.exec_())