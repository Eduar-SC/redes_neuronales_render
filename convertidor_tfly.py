import tensorflow as tf

model = tf.keras.models.load_model('brain_tumor_cnn.h5', compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('brain_tumor_cnn.tflite', 'wb') as f:
    f.write(tflite_model)