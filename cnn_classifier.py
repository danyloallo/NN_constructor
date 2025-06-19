import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, Callback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QListWidget, QListWidgetItem, QCheckBox,
                             QSpinBox, QLabel, QGroupBox, QFormLayout, QProgressBar, QComboBox,
                             QDoubleSpinBox, QMessageBox)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ProgressCallback(Callback):
    def __init__(self, progress_bar):
        super().__init__()
        self.progress_bar = progress_bar

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.setValue(epoch + 1)

class CNNClassifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Image Classifier")
        self.setGeometry(100, 100, 800, 600)
        self.dataset_dir = ""
        self.class_names = []
        self.selected_classes = []
        self.model = None
        self.history = None
        self.input_shape = (224, 224, 3)
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
        self.select_dataset_btn.clicked.connect(self.select_dataset)
        dataset_layout.addWidget(self.select_dataset_btn)
        self.class_list = QListWidget()
        self.class_list.setEnabled(False)
        dataset_layout.addWidget(QLabel("Выберите классы:"))
        dataset_layout.addWidget(self.class_list)
        dataset_group.setLayout(dataset_layout)
        controls_layout.addWidget(dataset_group)

        # Model configuration
        model_group = QGroupBox("Настройка модели")
        model_layout = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Ручная настройка", "Автоматическая (VGG16)"])
        self.mode_combo.currentIndexChanged.connect(self.toggle_model_config)
        model_layout.addRow("Тип настройки:", self.mode_combo)

        # Manual configuration widget
        self.manual_config = QWidget()
        manual_layout = QFormLayout()

        self.conv_blocks = QSpinBox()
        self.conv_blocks.setRange(1, 5)
        self.conv_blocks.setValue(2)
        manual_layout.addRow("Сверточные блоки:", self.conv_blocks)

        self.filters = []
        self.pooling_types = []
        self.pooling_sizes = []
        for i in range(5):
            filter_spin = QSpinBox()
            filter_spin.setRange(16, 512)
            filter_spin.setSingleStep(16)
            filter_spin.setValue(32 * (2 ** i))
            self.filters.append(filter_spin)
            manual_layout.addRow(f"Фильтров в блоке {i + 1}:", filter_spin)

            pooling_combo = QComboBox()
            pooling_combo.addItems(["MaxPooling2D", "AveragePooling2D"])
            self.pooling_types.append(pooling_combo)
            manual_layout.addRow(f"Тип пулинга {i + 1}:", pooling_combo)

            pooling_size_spin = QSpinBox()
            pooling_size_spin.setRange(2, 4)
            pooling_size_spin.setValue(2)
            self.pooling_sizes.append(pooling_size_spin)
            manual_layout.addRow(f"Размер пула {i + 1}:", pooling_size_spin)

        self.dense_units = QSpinBox()
        self.dense_units.setRange(32, 512)
        self.dense_units.setValue(128)
        manual_layout.addRow("Нейронов в полносвязном слое:", self.dense_units)

        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0.0, 0.5)
        self.dropout.setSingleStep(0.05)
        self.dropout.setValue(0.2)
        manual_layout.addRow("Dropout:", self.dropout)

        self.manual_config.setLayout(manual_layout)
        model_layout.addRow(self.manual_config)
        model_group.setLayout(model_layout)
        controls_layout.addWidget(model_group)

        # Training controls
        train_group = QGroupBox("Обучение")
        train_layout = QFormLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        train_layout.addRow("Эпохи:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 128)
        self.batch_spin.setValue(32)
        train_layout.addRow("Размер батча:", self.batch_spin)

        self.train_btn = QPushButton("Обучить модель")
        self.train_btn.clicked.connect(self.train_model)
        train_layout.addRow(self.train_btn)

        self.progress_bar = QProgressBar()
        train_layout.addRow(self.progress_bar)
        train_group.setLayout(train_layout)
        controls_layout.addWidget(train_group)

        # Model save/load
        io_group = QGroupBox("Сохранить или загрузить модель")
        io_layout = QHBoxLayout()
        self.save_btn = QPushButton("Сохранить модель")
        self.save_btn.clicked.connect(self.save_model)
        io_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Загрузить модель")
        self.load_btn.clicked.connect(self.load_model)
        io_layout.addWidget(self.load_btn)
        io_group.setLayout(io_layout)
        controls_layout.addWidget(io_group)

        # Prediction
        predict_group = QGroupBox("Классификация")
        predict_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Классифицировать изображение")
        self.predict_btn.clicked.connect(self.predict_image)
        predict_layout.addWidget(self.predict_btn)

        self.predict_label = QLabel("Классифицировано: None")
        predict_layout.addWidget(self.predict_label)
        predict_group.setLayout(predict_layout)
        controls_layout.addWidget(predict_group)

        main_layout.addWidget(controls_widget)

        # Right panel: Plots
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

    def toggle_model_config(self):
        self.manual_config.setVisible(self.mode_combo.currentText() == "Ручная настройка")

    def select_dataset(self):
        self.dataset_dir = QFileDialog.getExistingDirectory(self, "Выберите папку датасета")
        if self.dataset_dir:
            self.class_names = [d for d in os.listdir(self.dataset_dir)
                                if os.path.isdir(os.path.join(self.dataset_dir, d))]
            if not self.class_names:
                QMessageBox.warning(self, "Ошибка", "В папке датасета нет подпапок с классами!")
                return
            self.class_list.clear()
            self.selected_classes = []
            for class_name in self.class_names[:10]:  # Ограничение до 10 классов
                item = QListWidgetItem()
                checkbox = QCheckBox(class_name)
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
        if len(self.selected_classes) > 10:
            self.sender().setChecked(False)
            QMessageBox.warning(self, "Ошибка", "Выберите не более 10 классов")

    def build_manual_model(self, input_shape, num_classes):
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        num_blocks = self.conv_blocks.value()
        for i in range(num_blocks):
            filters = self.filters[i].value()
            model.add(layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
            pooling_type = self.pooling_types[i].currentText()
            pool_size = (self.pooling_sizes[i].value(), self.pooling_sizes[i].value())
            if pooling_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D(pool_size=pool_size))
            else:
                model.add(layers.AveragePooling2D(pool_size=pool_size))
        model.add(layers.Flatten())
        model.add(layers.Dense(self.dense_units.value(), activation='relu'))
        model.add(layers.Dropout(self.dropout.value()))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

    def build_auto_model(self, input_shape, num_classes):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False
        model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def train_model(self):
        if not self.selected_classes or len(self.selected_classes) < 2:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы 2 класса")
            return
        if not self.dataset_dir or not os.path.exists(self.dataset_dir):
            QMessageBox.warning(self, "Ошибка", "Ошибка при выборе датасета")
            return

        # Check for images
        for class_name in self.selected_classes:
            class_dir = os.path.join(self.dataset_dir, class_name)
            if not os.path.exists(class_dir):
                QMessageBox.warning(self, "Ошибка", f"Директория для класса {class_name} не существует")
                return
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                QMessageBox.warning(self, "Ошибка", f"В директории класса {class_name} нет изображений")
                return

        # Data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        try:
            train_generator = train_datagen.flow_from_directory(
                directory=self.dataset_dir,
                target_size=(224, 224),
                batch_size=self.batch_spin.value(),
                class_mode='categorical',
                subset='training',
                classes=self.selected_classes
            )
            validation_generator = train_datagen.flow_from_directory(
                directory=self.dataset_dir,
                target_size=(224, 224),
                batch_size=self.batch_spin.value(),
                class_mode='categorical',
                subset='validation',
                classes=self.selected_classes
            )
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить изображения: {str(e)}")
            return

        if train_generator.samples == 0 or validation_generator.samples == 0:
            QMessageBox.warning(self, "Ошибка", "Не найдено изображений для обучения или валидации")
            return

        # Build model
        num_classes = len(self.selected_classes)
        if self.mode_combo.currentText() == "Автоматическая (VGG16)":
            self.model = self.build_auto_model(self.input_shape, num_classes)
        else:
            self.model = self.build_manual_model(self.input_shape, num_classes)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train
        self.progress_bar.setRange(0, self.epochs_spin.value())
        early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=3, mode='max',
                                       restore_best_weights=True, verbose=1)
        try:
            self.history = self.model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=self.epochs_spin.value(),
                callbacks=[early_stopping, ProgressCallback(self.progress_bar)],
                verbose=1
            )
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка обучения: {str(e)}")
            return

        # Plot results
        self.plot_results()

    def plot_results(self):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend()
        self.canvas.draw()

        # Confusion matrix
        validation_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
            directory=self.dataset_dir,
            target_size=(224, 224),
            batch_size=self.batch_spin.value(),
            class_mode='categorical',
            subset='validation',
            classes=self.selected_classes
        )
        validation_generator.reset()
        y_pred = self.model.predict(validation_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = validation_generator.classes
        cm = confusion_matrix(y_true, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.selected_classes)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix')
        self.canvas.draw()

    def save_model(self):
        if self.model:
            save_path = QFileDialog.getSaveFileName(self, "Сохранить модель", "", "H5 Files (*.h5)")[0]
            if save_path:
                self.model.save(save_path)
                self.predict_label.setText("Модель успешно сохранена!")

    def load_model(self):
        load_path = QFileDialog.getOpenFileName(self, "Загрузить модель", "", "H5 Files (*.h5)")[0]
        if load_path:
            self.model = tf.keras.models.load_model(load_path)
            self.predict_label.setText("Модель успешно загружена!")

    def predict_image(self):
        if not self.model or not self.selected_classes:
            self.predict_label.setText("Нет загруженной модели или не выбраны классы!")
            return
        file_name = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.png *.jpg *.jpeg)")[0]
        if file_name:
            img = tf.keras.preprocessing.image.load_img(file_name, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) / 255.0
            predictions = self.model.predict(img_array)
            predicted_class = self.selected_classes[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            self.predict_label.setText(f"Классифицировано: {predicted_class} (уверенность: {confidence:.2%})")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CNNClassifierWindow()
    window.show()
    sys.exit(app.exec_())