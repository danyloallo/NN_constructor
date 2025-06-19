import sys
import pandas as pd
import tensorflow as tf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QListWidget, QListWidgetItem, QAbstractItemView, QMessageBox,
    QLineEdit, QFormLayout, QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QGroupBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
import pydot
from catboost import CatBoostRegressor
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import io

class PerceptronDataSelection(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Выбор данных для обучения перцептрона")
        self.setGeometry(100, 100, 600, 400)
        self.df = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Dataset selection
        dataset_group = QGroupBox("Выбор датасета")
        dataset_layout = QVBoxLayout()
        self.load_button = QPushButton("Загрузить CSV файл")
        self.load_button.setToolTip("Выберите CSV-файл с данными для обучения.\nРекомендация: Файл должен содержать числовые столбцы.")
        self.load_button.clicked.connect(self.load_csv)
        dataset_layout.addWidget(self.load_button)

        self.feature_label = QLabel("Выберите нецелевые признаки (X):")
        self.feature_label.setToolTip("Выберите столбцы, которые будут использоваться как входные данные.")
        dataset_layout.addWidget(self.feature_label)

        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.feature_list.setToolTip("Выберите нецелевые признаки (X) для модели.\nРекомендация: Выберите несколько столбцов с числовыми данными.")
        dataset_layout.addWidget(self.feature_list)

        self.target_label = QLabel("Выберите целевой признак (y):")
        self.target_label.setToolTip("Выберите столбец, который будет предсказываться моделью.")
        dataset_layout.addWidget(self.target_label)

        self.target_list = QListWidget()
        self.target_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.target_list.setToolTip("Выберите один целевой признак (y).\nРекомендация: Выберите столбец с числовыми значениями для регрессии.")
        dataset_layout.addWidget(self.target_list)
        dataset_group.setLayout(dataset_layout)
        main_layout.addWidget(dataset_group)

        # Navigation buttons
        button_group = QGroupBox("Настройка модели")
        button_layout = QHBoxLayout()
        self.auto_button = QPushButton("Автоматическая настройка")
        self.auto_button.setToolTip("Перейти к автоматической настройке с использованием CatBoost.")
        self.auto_button.clicked.connect(self.go_to_auto_setup)
        button_layout.addWidget(self.auto_button)

        self.manual_button = QPushButton("Ручная настройка")
        self.manual_button.setToolTip("Перейти к ручной настройке многослойного перцептрона.")
        self.manual_button.clicked.connect(self.go_to_manual_setup)
        button_layout.addWidget(self.manual_button)
        button_group.setLayout(button_layout)
        main_layout.addWidget(button_group)

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV файл", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.feature_list.clear()
                self.target_list.clear()
                for col in self.df.columns:
                    self.feature_list.addItem(QListWidgetItem(col))
                    self.target_list.addItem(QListWidgetItem(col))
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{str(e)}")

    def validate_selection(self):
        features = [item.text() for item in self.feature_list.selectedItems()]
        target_items = self.target_list.selectedItems()
        if not features or not target_items:
            QMessageBox.warning(self, "Внимание", "Выберите нецелевые и целевой признаки!")
            return None, None
        return features, target_items[0].text()

    def go_to_auto_setup(self):
        features, target = self.validate_selection()
        if features and target:
            self.auto_window = AutoSetupWindow(self.df, features, target)
            self.auto_window.show()
            self.hide()

    def go_to_manual_setup(self):
        features, target = self.validate_selection()
        if features and target:
            self.manual_window = ManualSetupWindow(self.df, features, target)
            self.manual_window.show()
            self.hide()

class AutoSetupWindow(QMainWindow):
    def __init__(self, df, features, target):
        super().__init__()
        self.setWindowTitle("Автоматическая настройка модели")
        self.setGeometry(100, 100, 800, 600)
        self.df = df
        self.features = features
        self.target = target
        self.model = None
        self.X_test = None
        self.y_test = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Input fields
        input_group = QGroupBox("Ввод данных для предсказания")
        input_layout = QFormLayout()
        self.input_fields = {}
        for feature in self.features:
            field = QLineEdit()
            field.setPlaceholderText(f"Введите значение для {feature}")
            field.setToolTip(f"Введите числовое значение для признака {feature}.\nРекомендация: Вводите значения в том же диапазоне, что в обучающих данных.")
            input_layout.addRow(feature, field)
            self.input_fields[feature] = field
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # Training controls
        train_group = QGroupBox("Обучение")
        train_layout = QHBoxLayout()
        self.train_button = QPushButton("Обучить модель")
        self.train_button.setToolTip("Обучить модель CatBoost с автоматическими параметрами.")
        self.train_button.clicked.connect(self.train_model)
        train_layout.addWidget(self.train_button)

        self.predict_button = QPushButton("Сделать предсказание")
        self.predict_button.setToolTip("Сделать предсказание для введённых значений признаков.")
        self.predict_button.clicked.connect(self.make_prediction)
        train_layout.addWidget(self.predict_button)
        train_group.setLayout(train_layout)
        main_layout.addWidget(train_group)

        # Model save
        save_group = QGroupBox("Сохранение модели")
        save_layout = QHBoxLayout()
        self.save_button = QPushButton("Сохранить модель")
        self.save_button.setToolTip("Сохранить обученную модель в файл .pkl.")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        save_layout.addWidget(self.save_button)
        save_group.setLayout(save_layout)
        main_layout.addWidget(save_group)

    def train_model(self):
        try:
            X = self.df[self.features]
            y = self.df[self.target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.X_test = X_test
            self.y_test = y_test

            self.model = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=10)
            self.model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)
            self.save_button.setEnabled(True)

            self.show_results_in_new_window()
            QMessageBox.information(self, "Успех", "Модель успешно обучена!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обучении:\n{str(e)}")

    def show_results_in_new_window(self):
        # Графики в новом окне
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        if self.model:
            train_loss = self.model.get_evals_result()['learn']['RMSE']
            test_loss = self.model.get_evals_result()['validation']['RMSE']
            ax1.plot(train_loss, label='Train RMSE')
            ax1.plot(test_loss, label='Test RMSE')
            ax1.set_title('Ошибка обучения')
            ax1.set_xlabel('Итерация')
            ax1.set_ylabel('RMSE')
            ax1.legend()

        if self.X_test is not None and self.y_test is not None:
            y_pred = self.model.predict(self.X_test)
            ax2.scatter(self.y_test, y_pred, alpha=0.5)
            min_val = min(min(self.y_test), min(y_pred))
            max_val = max(max(self.y_test), max(y_pred))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Идеальное соответствие')
            ax2.set_title('Предсказанные vs Истинные (тестовая выборка)')
            ax2.set_xlabel('Истинные значения')
            ax2.set_ylabel('Предсказанные значения')
            ax2.legend()

        fig.tight_layout()
        plt.savefig('training_results.png')
        plt.show()

        # Таблица метрик в новом окне
        metric_window = QWidget()
        metric_window.setWindowTitle("Метрики обучения")
        metric_window.setGeometry(150, 150, 400, 300)
        layout = QVBoxLayout()
        table = QTableWidget()
        if self.model:
            train_loss = self.model.get_evals_result()['learn']['RMSE']
            test_loss = self.model.get_evals_result()['validation']['RMSE']
            table.setRowCount(len(train_loss))
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["Итерация", "Train RMSE", "Test RMSE"])
            for i in range(len(train_loss)):
                table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                table.setItem(i, 1, QTableWidgetItem(f"{train_loss[i]:.4f}"))
                table.setItem(i, 2, QTableWidgetItem(f"{test_loss[i]:.4f}"))
        table.resizeColumnsToContents()
        layout.addWidget(table)
        metric_window.setLayout(layout)
        metric_window.show()
        self.metric_window = metric_window

    def make_prediction(self):
        if not self.model:
            QMessageBox.warning(self, "Внимание", "Сначала обучите модель!")
            return
        try:
            input_data = []
            for feature in self.features:
                value = self.input_fields[feature].text()
                if not value:
                    QMessageBox.warning(self, "Внимание", f"Введите значение для {feature}")
                    return
                input_data.append(float(value))
            prediction = self.model.predict([input_data])
            QMessageBox.information(self, "Предсказание", f"Предсказанное значение: {prediction[0]:.4f}")
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Введите числовые значения!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при предсказании:\n{str(e)}")

    def save_model(self):
        if not self.model:
            QMessageBox.warning(self, "Внимание", "Сначала обучите модель!")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", "", "Pickle Files (*.pkl)")
        if file_path:
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(self.model, f)
                QMessageBox.information(self, "Успех", "Модель успешно сохранена!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении:\n{str(e)}")

class ManualSetupWindow(QMainWindow):
    def __init__(self, df, features, target):
        super().__init__()
        self.setWindowTitle("Ручная настройка перцептрона")
        self.setGeometry(100, 100, 800, 600)
        self.df = df
        self.features = features
        self.target = target
        self.model = None
        self.scaler = StandardScaler()
        self.X_test = None
        self.y_test = None
        self.train_loss_history = []
        self.val_loss_history = []
        self.neuron_inputs = []
        self.activation_combos = []
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Model configuration
        model_group = QGroupBox("Настройка модели")
        self.config_layout = QFormLayout()
        self.num_layers_combo = QComboBox()
        self.num_layers_combo.addItems([str(i) for i in range(1, 6)])
        self.num_layers_combo.setToolTip("Количество скрытых слоёв в перцептроне.\nРекомендация: 1–3 слоя для большинства задач.")
        self.num_layers_combo.currentIndexChanged.connect(self.update_neuron_inputs)
        self.config_layout.addRow("Количество слоев:", self.num_layers_combo)

        # Поля для нейронов и активаций
        self.neuron_layout = QFormLayout()
        self.update_neuron_inputs()
        self.config_layout.addRow("Нейроны и активации:", self.neuron_layout)

        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0001, 0.1)
        self.learning_rate_input.setSingleStep(0.001)
        self.learning_rate_input.setValue(0.01)
        self.learning_rate_input.setToolTip("Скорость обучения модели.\nРекомендация: 0.001–0.01 для стабильного обучения.")
        self.config_layout.addRow("Скорость обучения:", self.learning_rate_input)

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(10, 1000)
        self.epochs_input.setValue(500)
        self.epochs_input.setToolTip("Количество эпох обучения.\nРекомендация: 100–500 для небольших датасетов.")
        self.config_layout.addRow("Количество эпох:", self.epochs_input)
        model_group.setLayout(self.config_layout)
        main_layout.addWidget(model_group)

        # Input fields
        input_group = QGroupBox("Ввод данных для предсказания")
        input_layout = QFormLayout()
        self.input_fields = {}
        for feature in self.features:
            field = QLineEdit()
            field.setPlaceholderText(f"Введите значение для {feature}")
            field.setToolTip(f"Введите числовое значение для признака {feature}.\nРекомендация: Вводите значения в том же диапазоне, что в обучающих данных.")
            input_layout.addRow(feature, field)
            self.input_fields[feature] = field
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # Training controls
        train_group = QGroupBox("Обучение")
        train_layout = QHBoxLayout()
        self.train_button = QPushButton("Обучить модель")
        self.train_button.setToolTip("Обучить перцептрон с заданными параметрами.")
        self.train_button.clicked.connect(self.train_model)
        train_layout.addWidget(self.train_button)

        self.predict_button = QPushButton("Сделать предсказание")
        self.predict_button.setToolTip("Сделать предсказание для введённых значений признаков.")
        self.predict_button.clicked.connect(self.make_prediction)
        train_layout.addWidget(self.predict_button)
        train_group.setLayout(train_layout)
        main_layout.addWidget(train_group)

        # Model save and architecture
        save_group = QGroupBox("Сохранение и архитектура")
        save_layout = QHBoxLayout()
        self.save_button = QPushButton("Сохранить модель")
        self.save_button.setToolTip("Сохранить обученную модель и скейлер в файл .keras.")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        save_layout.addWidget(self.save_button)

        self.show_arch_button = QPushButton("Показать архитектуру")
        self.show_arch_button.setToolTip("Отобразить архитектуру нейросети в виде графа.")
        self.show_arch_button.clicked.connect(self.show_architecture)
        save_layout.addWidget(self.show_arch_button)
        save_group.setLayout(save_layout)
        main_layout.addWidget(save_group)

    def update_neuron_inputs(self):
        # Полностью очищаем neuron_layout
        while self.neuron_layout.count():
            item = self.neuron_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Рекурсивно очищаем под-лейауты
                while item.layout().count():
                    sub_item = item.layout().takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().deleteLater()
                item.layout().deleteLater()
        self.neuron_inputs.clear()
        self.activation_combos.clear()

        num_layers = int(self.num_layers_combo.currentText())
        for i in range(num_layers):
            neuron_input = QSpinBox()
            neuron_input.setRange(1, 100)
            neuron_input.setValue(32)
            neuron_input.setToolTip(f"Количество нейронов в скрытом слое {i + 1}.\nРекомендация: 16–64 нейронов на слой.")
            self.neuron_inputs.append(neuron_input)

            activation_combo = QComboBox()
            activation_combo.addItems(['relu', 'tanh', 'sigmoid', 'linear'])
            activation_combo.setCurrentText('relu')
            activation_combo.setToolTip(f"Функция активации для слоя {i + 1}.\nРекомендация: Используйте relu для большинства задач.")
            self.activation_combos.append(activation_combo)

            # Добавляем поля в neuron_layout
            self.neuron_layout.addRow(f"Нейронов в слое {i + 1}:", neuron_input)
            self.neuron_layout.addRow(f"Активация слоя {i + 1}:", activation_combo)

    def create_network_graph(self):
        graph = pydot.Dot(graph_type='digraph', rankdir='LR', splines='line')
        input_node = pydot.Node(f"Input\n{len(self.features)} neurons", shape="box")
        graph.add_node(input_node)

        hidden_layers = [int(input.value()) for input in self.neuron_inputs]
        activations = [combo.currentText().capitalize() for combo in self.activation_combos]
        prev_node = input_node
        for i, (neurons, activation) in enumerate(zip(hidden_layers, activations), 1):
            layer_node = pydot.Node(f"Layer {i}\n{neurons} neurons\n{activation}", shape="box")
            graph.add_node(layer_node)
            edge = pydot.Edge(prev_node, layer_node)
            graph.add_edge(edge)
            prev_node = layer_node

        output_node = pydot.Node("Output\n1 neuron", shape="box")
        graph.add_node(output_node)
        edge = pydot.Edge(prev_node, output_node)
        graph.add_edge(edge)

        svg_data = graph.create_svg()
        return svg_data

    def show_architecture(self):
        try:
            svg_data = self.create_network_graph()
            pixmap = QPixmap()
            pixmap.loadFromData(svg_data, "SVG")
            scaled_pixmap = pixmap.scaled(400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            arch_window = QWidget()
            arch_window.setWindowTitle("Архитектура нейросети")
            arch_window.setGeometry(150, 150, 400, 200)
            layout = QVBoxLayout()
            arch_label = QLabel()
            arch_label.setPixmap(scaled_pixmap)
            arch_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(arch_label)
            arch_window.setLayout(layout)
            arch_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось отобразить архитектуру:\n{str(e)}")

    def train_model(self):
        try:
            X = self.df[self.features]
            y = self.df[self.target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.X_test = X_test
            self.y_test = y_test

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            hidden_layer_sizes = [int(input.value()) for input in self.neuron_inputs]
            activations = [combo.currentText() for combo in self.activation_combos]
            learning_rate = self.learning_rate_input.value()
            max_iter = self.epochs_input.value()

            # Построение модели Keras
            self.model = models.Sequential()
            self.model.add(layers.Input(shape=(len(self.features),)))
            for neurons, activation in zip(hidden_layer_sizes, activations):
                self.model.add(layers.Dense(neurons, activation=activation))
            self.model.add(layers.Dense(1))  # Выходной слой для регрессии
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

            history = self.model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                                     epochs=max_iter, batch_size=32, verbose=1)
            self.train_loss_history = history.history['loss']
            self.val_loss_history = history.history['val_loss']
            self.save_button.setEnabled(True)

            self.show_results_in_new_window()
            self.show_architecture()
            QMessageBox.information(self, "Успех", f"Модель обучена! Слои: {hidden_layer_sizes}, Активации: {activations}")
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка ввода: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обучении:\n{str(e)}")

    def show_results_in_new_window(self):
        # Графики в новом окне
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        if self.train_loss_history:
            ax1.plot(self.train_loss_history, label='Train Loss (MSE)')
            ax1.plot(self.val_loss_history, label='Validation Loss (MSE)')
            ax1.set_title('Ошибка обучения')
            ax1.set_xlabel('Эпоха')
            ax1.set_ylabel('MSE')
            ax1.legend()

        if self.model and self.X_test is not None and self.y_test is not None:
            X_test_scaled = self.scaler.transform(self.X_test)
            y_pred = self.model.predict(X_test_scaled, verbose=0).flatten()
            ax2.scatter(self.y_test, y_pred, alpha=0.5)
            min_val = min(min(self.y_test), min(y_pred))
            max_val = max(max(self.y_test), max(y_pred))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Идеальное соответствие')
            ax2.set_title('Предсказанные vs Истинные (тестовая выборка)')
            ax2.set_xlabel('Истинные значения')
            ax2.set_ylabel('Предсказанные значения')
            ax2.legend()

        fig.tight_layout()
        plt.savefig('training_results.png')
        plt.show()

        # Таблица метрик в новом окне
        metric_window = QWidget()
        metric_window.setWindowTitle("Метрики обучения")
        metric_window.setGeometry(150, 150, 400, 300)
        layout = QVBoxLayout()
        table = QTableWidget()
        table.setRowCount(len(self.train_loss_history))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Эпоха", "Train Loss (MSE)", "Validation Loss (MSE)"])
        for i in range(len(self.train_loss_history)):
            table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            table.setItem(i, 1, QTableWidgetItem(f"{self.train_loss_history[i]:.4f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{self.val_loss_history[i]:.4f}"))
        table.resizeColumnsToContents()
        layout.addWidget(table)
        metric_window.setLayout(layout)
        metric_window.show()
        self.metric_window = metric_window

    def make_prediction(self):
        if not self.model:
            QMessageBox.warning(self, "Внимание", "Сначала обучите модель!")
            return
        try:
            input_data = []
            for feature in self.features:
                value = self.input_fields[feature].text()
                if not value:
                    QMessageBox.warning(self, "Внимание", f"Введите значение для {feature}")
                    return
                input_data.append(float(value))
            input_data_scaled = self.scaler.transform([input_data])
            prediction = self.model.predict(input_data_scaled, verbose=0)[0][0]
            QMessageBox.information(self, "Предсказание", f"Предсказанное значение: {prediction:.4f}")
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Введите числовые значения!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при предсказании:\n{str(e)}")

    def save_model(self):
        if not self.model:
            QMessageBox.warning(self, "Внимание", "Сначала обучите модель!")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", "", "Keras Files (*.keras)")
        if file_path:
            try:
                self.model.save(file_path)
                scaler_path = file_path.replace('.keras', '_scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                QMessageBox.information(self, "Успех", "Модель и скейлер успешно сохранены!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PerceptronDataSelection()
    window.show()
    sys.exit(app.exec_())