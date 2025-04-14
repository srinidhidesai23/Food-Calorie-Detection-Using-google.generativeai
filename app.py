from flask import Flask, request, render_template, redirect, url_for
import os
import math
import csv
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import PIL
import google.generativeai as genai

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Your Gemini API Key
os.environ['GOOGLE_API_KEY'] = "AIzaSyCE75L9ZJi7fzSghw6D_U7psANn6des5I8"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

label = ['apple pie', 'baby back ribs', 'baklava', 'beef carpaccio', 'beef tartare', 'beet salad', 'beignets', 'bibimbap', 'bread pudding', 'breakfast burrito', 'bruschetta', 'caesar salad', 'cannoli', 'caprese salad', 'carrot cake', 'ceviche', 'cheese plate', 'cheesecake', 'chicken curry', 'chicken quesadilla', 'chicken wings', 'chocolate cake', 'chocolate mousse', 'churros', 'clam chowder', 'club sandwich', 'crab cakes', 'creme brulee', 'croque madame', 'cup cakes', 'deviled eggs', 'donuts', 'dumplings', 'edamame', 'eggs benedict', 'escargots', 'falafel', 'filet mignon', 'fish and_chips', 'foie gras', 'french fries', 'french onion soup', 'french toast', 'fried calamari', 'fried rice', 'frozen yogurt', 'garlic bread', 'gnocchi', 'greek salad', 'grilled cheese sandwich', 'grilled salmon', 'guacamole', 'gyoza', 'hamburger', 'hot and sour soup', 'hot dog', 'huevos rancheros', 'hummus', 'ice cream', 'lasagna', 'lobster bisque', 'lobster roll sandwich', 'macaroni and cheese', 'macarons', 'miso soup', 'mussels', 'nachos', 'omelette', 'onion rings', 'oysters', 'pad thai', 'paella', 'pancakes', 'panna cotta', 'peking duck', 'pho', 'pizza', 'pork chop', 'poutine', 'prime rib', 'pulled pork sandwich', 'ramen', 'ravioli', 'red velvet cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed salad', 'shrimp and grits', 'spaghetti bolognese', 'spaghetti carbonara', 'spring rolls', 'steak', 'strawberry shortcake', 'sushi', 'tacos', 'octopus balls', 'tiramisu', 'tuna tartare', 'waffles']
label.sort()

nu_link = 'https://www.nutritionix.com/food/'

tensorflow.keras.backend.clear_session()
model_best = load_model('best_model_101class.keras', compile=False)
print('✅ Model successfully loaded!')

start = [0]
passed = [0]
pack = [[]]
num = [0]

nutrients = [
    {'name': 'protein', 'value': 0.0},
    {'name': 'calcium', 'value': 0.0},
    {'name': 'fat', 'value': 0.0},
    {'name': 'carbohydrates', 'value': 0.0},
    {'name': 'vitamins', 'value': 0.0}
]

with open('nutrition101.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    nutrition_table = {}
    for row in reader:
        name = row[1].strip()
        nutrition_table[name] = [
            {'name': 'protein', 'value': float(row[2])},
            {'name': 'calcium', 'value': float(row[3])},
            {'name': 'fat', 'value': float(row[4])},
            {'name': 'carbohydrates', 'value': float(row[5])},
            {'name': 'vitamins', 'value': float(row[6])}
        ]

@app.route('/')
def index():
    return render_template('index.html', img='static/profile.jpg')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html', img='static/profile.jpg')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist("img")
    if not files or files[0].filename == '':
        return "⚠️ No file selected."

    for f in files:
        filename = secure_filename(str(num[0] + 500) + '.jpg')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        num[0] += 1

    pack[0] = []
    return redirect(url_for('predict'))

@app.route('/predict')
def predict():
    if num[0] == 0:
        return "⚠️ No images uploaded yet."

    print(f'📸 Total images: {num[0]}')

    for i in range(start[0], num[0]):
        pa = {}
        filename = f'{UPLOAD_FOLDER}/{i + 500}.jpg'

        try:
            img = image.load_img(filename, target_size=(224, 224))
            pred_img = image.img_to_array(img)
            pred_img = np.expand_dims(pred_img, axis=0)
            pred_img = pred_img / 255.0
        except Exception as e:
            print(f"⚠️ Error loading image {filename}: {e}")
            continue

        pred = model_best.predict(pred_img)
        top = pred.argsort()[0][-3:]

        _true = label[top[2]]
        pa['image'] = filename
        pa['result'] = {
            _true: float("{:.2f}".format(pred[0][top[2]] * 100)),
            label[top[1]]: float("{:.2f}".format(pred[0][top[1]] * 100)),
            label[top[0]]: float("{:.2f}".format(pred[0][top[0]] * 100))
        }
        pa['nutrition'] = nutrition_table.get(_true, [])
        pa['food'] = f'{nu_link}{_true}'
        pa['idx'] = i - start[0]
        pa['quantity'] = 100

        vis_img = PIL.Image.open(filename)

        # ✅ Updated Gemini model
        vision_model = genai.GenerativeModel('gemini-1.5-flash')
        pa['ai'] = vision_model.generate_content([
            "Give me response in this form: 'The image displays /items/ with an /estimated calories for all items and give accurate calories /'",
            vis_img
        ])

        pack[0].append(pa)
        passed[0] += 1

    start[0] = passed[0]

    for p in pack[0]:
        for j in range(5):
            nutrients[j]['value'] += p['nutrition'][j]['value']

    for j in range(5):
        nutrients[j]['value'] /= num[0]

    print("✅ Nutrition data processed.")
    return render_template('results.html', pack=pack[0], whole_nutrition=nutrients)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
