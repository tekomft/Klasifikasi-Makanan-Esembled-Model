from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from datetime import datetime
from flask_cors import CORS
import pymysql
pymysql.install_as_MySQLdb()
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import time
import os

app = Flask(__name__)
CORS(app)

# Konfigurasi database MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/food_db'
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
modelMobileNetV2 = load_model(r"D:\Kuliah Semester 6\Jurnal\Materi\CNN\Codingan\Model_Skripsi\mobilenetv2_Percobaan 4(2).h5")
modelXception = load_model(r"D:\Kuliah Semester 6\Jurnal\Materi\CNN\Codingan\Model_Skripsi\Percobaan3_Xception.h5")
modelEfficientNetB0 = load_model(r"D:\Kuliah Semester 6\Jurnal\Materi\CNN\Codingan\Model_Skripsi\Percobaan3(2)_EfficientNetB0.h5")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("cnn_db.html")

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("classifications_db.html")

# Rute untuk mendapatkan semua resep dari database
@app.route('/recipes', methods=['GET'])
def get_recipes():
    recipes = food_recipe.query.all()
    return jsonify([{
        'id': recipe.id,
        'food_name': recipe.food_name,
        'ingredients': recipe.ingredients,
        'recipe': recipe.recipe
    } for recipe in recipes])

# Rute untuk menambahkan resep ke database
@app.route('/add_recipe', methods=['POST'])
def add_recipe():
    data = request.json
    new_recipe = food_recipe(
        food_name=data['food_name'],
        ingredients=data['ingredients'],
        recipe=data['recipe']
    )
    try:
        db.session.add(new_recipe)
        db.session.commit()
        return jsonify({'message': 'Recipe added successfully!'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp

    files = request.files.getlist('file')
    filename = "temp_image.png"
    errors = {}
    success = False
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors["message"] = f'File type of {file.filename} is not allowed'

    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp

    img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Convert image to RGB
    img = Image.open(img_url).convert('RGB')
    now = datetime.now()
    predict_image_path = 'static/uploads/' + now.strftime("%d%m%y-%H%M%S") + ".png"
    img.convert('RGB').save(predict_image_path, format="png")
    img.close()

    # Prepare image for prediction
    img = image.load_img(predict_image_path, target_size=(224, 224, 3))
    x = image.img_to_array(img)
    x = x / 127.5 - 1
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    #Mulai menghitung TE
    start_time = time.time()

    # Predict
    prediction_array_MobileNetV2 = modelMobileNetV2.predict(images)
    prediction_array_Xception = modelXception.predict(images)
    prediction_array_EfficientNetB0 = modelEfficientNetB0.predict(images)

    #Selesai menghitung TE
    end_time = time.time()
    execution_time = round(end_time - start_time, 3)  # Waktu dalam detik

    # Prepare class names and confidence threshold
    class_names = ['Buras', 'Dangkot', 'Gogos', 'Kapurung', 'Sokko', 'Sop Konro']
    confidence_threshold = 0.5

    def get_prediction(prediction_array):
        if np.max(prediction_array) < confidence_threshold:
            return "gambar tidak terdapat pada kelas", "0%"
        else:
            return class_names[np.argmax(prediction_array)], '{:2.0f}%'.format(100 * np.max(prediction_array))

    # Get predictions for each model
    prediction_mobilenetv2, confidence_mobilenetv2 = get_prediction(prediction_array_MobileNetV2)
    prediction_xception, confidence_xception = get_prediction(prediction_array_Xception)
    prediction_efficientnetb0, confidence_efficientnetb0 = get_prediction(prediction_array_EfficientNetB0)

    # Menggabungkan prediksi dari semua model
    predictions = [
         (prediction_mobilenetv2, np.max(prediction_array_MobileNetV2)),
         (prediction_xception, np.max(prediction_array_Xception)),
         (prediction_efficientnetb0, np.max(prediction_array_EfficientNetB0))
     ]

    # Pilih prediksi dengan confidence tertinggi
    predicted_food, confidence = get_prediction(prediction_array_MobileNetV2)

    # Mencari resep berdasarkan nama makanan yang terdeteksi
    recipe_details = food_recipe.query.filter_by(food_name=predicted_food).first()
    if recipe_details:
        recipe_data = {
            'food_name': recipe_details.food_name,
            'ingredients': recipe_details.ingredients,
            'recipe': recipe_details.recipe
        }
    else:
        recipe_data = {'error': 'Resep tidak ditemukan'}

    # Mengembalikan data hasil klasifikasi dan resep
    return render_template("classifications_db.html", 
                           img_path=predict_image_path,
                            predictionmobilenetv2=prediction_mobilenetv2,
                            confidencemobilenetv2=confidence_mobilenetv2,
                            predictionxception=prediction_xception,
                            confidencexception=confidence_xception,
                            predictionefficientnetb0=prediction_efficientnetb0,
                            confidenceefficientnetb0=confidence_efficientnetb0,
                            execution_time=execution_time,
                            recipe_data=recipe_data)

if __name__ == '__main__':
    #with app.app_context():
    #    db.create_all() 
    app.run(debug=True)
