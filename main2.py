from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import shutil
import matplotlib.pyplot as plt

app = Flask(__name__)
# Diccionario de clases
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Cargar modelo entrenado pero de h5
# model = tf.keras.models.load_model('brain_tumor_cnn.h5')

# Carpeta de subida
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ------------------------------
# Cargar modelo TFLite
def load_tflite_model():
    model_path = 'brain_tumor_cnn.tflite'
    interpreter =tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

tflite_interpreter = load_tflite_model()
#-------------------------------
# Función para predecir usando el modelo TFLite
def predict_image_class(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    # Asegurarse de que la imagen tenga el tamaño correcto
    img_array = img_array.astype(np.float32)

    # Cargar el tensor de entrada
    interpreter.set_tensor(input_index, img_array)
    # Ejecutar la inferencia
    interpreter.invoke()

    # Obtener la predicción
    output_data = interpreter.get_tensor(output_index)
    return output_data[0]
#-------------------------------


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    probabilities = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Copiar imagen a carpeta static/uploads para previsualizar
            static_path = os.path.join('static', 'uploads', filename)
            os.makedirs(os.path.dirname(static_path), exist_ok=True)
            shutil.copy(filepath, static_path)

            # Preprocesar imagen
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predicción
            prediction = predict_image_class(tflite_interpreter,img_array)
            predicted_class = class_names[np.argmax(prediction)]
            prediction_result = f"Predicción: {predicted_class.upper()}"
            probabilities = {class_names[i]: float(f"{prob:.4f}") for i, prob in enumerate(prediction)}

            # Graficar probabilidades
            plt.figure(figsize=(6,4))
            plt.bar(probabilities.keys(), probabilities.values(), color='skyblue')
            plt.title('Probabilidades por clase')
            plt.ylabel('Confianza')
            plt.tight_layout()
            graph_path = os.path.join('static', 'uploads', 'probabilidades.png')
            plt.savefig(graph_path)
            plt.close()

    return render_template('index.html', prediction=prediction_result, image_name=filename, probs=probabilities)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)