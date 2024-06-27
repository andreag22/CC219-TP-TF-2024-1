from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
import numpy as np

app = Flask(__name__)

# Cargar los modelos al inicio
tl_model_path = 'eye_disease_classifier_tl.h5'
cnn_model_path = 'eye_disease_classifier_cnn.h5'

modelo_tl = load_model(tl_model_path)
modelo_cnn = load_model(cnn_model_path)

# Dimensiones esperadas por los modelos
img_height, img_width = 150, 150

# Define las etiquetas de las clases (para ambos modelos, asumiendo las mismas clases)
class_labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    print("Solicitud POST recibida en /predict")
    try:
        # Asegúrate de recibir la imagen desde la solicitud POST
        file = request.files['image']
        img = image.load_img(file, target_size=(img_height, img_width))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Preprocesamiento según el modelo seleccionado
        model_name = request.form['model']
        if model_name == 'tl':
            x = preprocess_input_vgg16(x)  # Ajusta según el preprocesamiento adecuado para tu modelo TL
            model = modelo_tl
        elif model_name == 'cnn':
            x = preprocess_input_resnet50(x)  # Ajusta según el preprocesamiento adecuado para tu modelo CNN
            model = modelo_cnn
        else:
            return jsonify({'error': 'Modelo no reconocido'})

        # Hacer la predicción
        predictions = model.predict(x)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_label = class_labels[predicted_class_index]

        # Devolver la respuesta como JSON
        response = {'prediction': predicted_class_label}
        print("Respuesta:", response)  # Verifica la estructura de la respuesta
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})  # Devuelve cualquier error como JSON

if __name__ == '__main__':
    app.run(debug=True)
