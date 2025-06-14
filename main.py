import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ✅ Rutas dinámicas correctas para cualquier sistema operativo
base_path = os.path.join(os.getcwd(), "dataset")
train_dir = os.path.join(base_path, "Training")
test_dir = os.path.join(base_path, "Testing")

print("Ruta entrenamiento:", train_dir)
print("Ruta prueba:", test_dir)
print("¿Existen?:", os.path.exists(train_dir), os.path.exists(test_dir))

# Generadores de datos
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

# Compilar y entrenar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, validation_data=test_gen, epochs=10)

# Guardar modelo
model.save("brain_tumor_cnn.h5")

# Gráficos de precisión y pérdida
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title("Precisión"), plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title("Pérdida"), plt.legend()
plt.tight_layout()
plt.savefig("training_result.png")
plt.close()

# Evaluación
test_gen.reset()
predictions = model.predict(test_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
matrix = confusion_matrix(true_classes, predicted_classes)

print(report)
print(matrix)