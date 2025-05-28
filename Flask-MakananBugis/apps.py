from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from datetime import datetime
from flask_cors import CORS
import pymysql
import numpy as np
import time
import os

pymysql.install_as_MySQLdb()
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
CORS(app)

# Konfigurasi database MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/food_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Definisi model database
class food_recipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    food_name = db.Column(db.String(50), unique=True, nullable=False)
    ingredients = db.Column(db.Text, nullable=False)
    recipe = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<food_recipe {self.food_name}>'

# Load models for prediction
try:
    modelMobileNetV2 = load_model(r"D:\Kuliah Semester 6\Jurnal\Materi\CNN\Codingan\Model_Skripsi\mobilenetv2_Percobaan 4(2).h5")
    modelXception = load_model(r"D:\Kuliah Semester 6\Jurnal\Materi\CNN\Codingan\Model_Skripsi\Percobaan3_Xception.h5")
    modelEfficientNetB0 = load_model(r"D:\Kuliah Semester 6\Jurnal\Materi\CNN\Codingan\Model_Skripsi\Percobaan3(2)_EfficientNetB0.h5")
except Exception as e:
    print(f"Error loading model: {e}")

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template("cnn_db.html")

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    return render_template("classifications.html")

@app.route('/recipes', methods=['GET'])
def get_recipes():
    recipes = food_recipe.query.all()
    return jsonify([{
        'id': recipe.id,
        'food_name': recipe.food_name,
        'ingredients': recipe.ingredients,
        'recipe': recipe.recipe
    } for recipe in recipes])

@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'message': 'No image in the request'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
        filename = f"image_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    else:
        return jsonify({'message': 'File type not allowed'}), 400

    img = Image.open(filepath).convert('RGB')
    predict_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{timestamp}.jpg")
    img.save(predict_image_path, format="JPEG")

    img = image.load_img(predict_image_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    start_time = time.time()

    try:
        pred_mobilenetv2 = modelMobileNetV2.predict(images)
        pred_xception = modelXception.predict(images)
        pred_efficientnetb0 = modelEfficientNetB0.predict(images)
    except Exception as e:
        return jsonify({'error': f'Error in model prediction: {e}'}), 500

    execution_time = round(time.time() - start_time, 3)

    class_names = ['Buras', 'Dangkot', 'Gogos', 'Kapurung', 'Sokko', 'Sop Konro']

    avg_prob = (pred_mobilenetv2 + pred_xception + pred_efficientnetb0) / 3
    predicted_index = np.argmax(avg_prob)
    predicted_food = class_names[predicted_index]
    confidence = f"{100 * avg_prob[0][predicted_index]:.2f}%"

    # Periksa apakah prediksi masuk dalam 6 kelas atau bukan
    if avg_prob[0][predicted_index] < 0.2:  # Jika confidence < 50%, dianggap tidak dikenali
        predicted_food = "Gambar tidak terdapat pada kelas"
        recipe_data = {'error': 'Resep tidak ditemukan'}
    else:
        recipe_details = food_recipe.query.filter_by(food_name=predicted_food.capitalize()).first()
        recipe_data = {
            'food_name': recipe_details.food_name,
            'ingredients': recipe_details.ingredients,
            'recipe': recipe_details.recipe
        } if recipe_details else {'error': 'Resep tidak ditemukan'}

    return render_template("classifications.html", 
                           img_path=predict_image_path,
                           predictionmobilenetv2=class_names[np.argmax(pred_mobilenetv2)], 
                           confidencemobilenetv2=f"{100 * np.max(pred_mobilenetv2):.2f}%",
                           predictionxception=class_names[np.argmax(pred_xception)], 
                           confidencexception=f"{100 * np.max(pred_xception):.2f}%",
                           predictionefficientnetb0=class_names[np.argmax(pred_efficientnetb0)], 
                           confidenceefficientnetb0=f"{100 * np.max(pred_efficientnetb0):.2f}%",
                           ensemble_prediction=predicted_food, 
                           ensemble_confidence=confidence,
                           execution_time=execution_time, 
                           recipe_data=recipe_data)

if __name__ == '__main__':
    app.run(debug=True)
