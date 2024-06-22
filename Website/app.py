from flask import Flask, request, render_template, redirect, url_for, jsonify, session
from flask_cors import CORS
import numpy as np
from keras.models import load_model
import pickle

app = Flask(__name__)
CORS(app)
app.secret_key = '123'

model_path = 'crop_recommendation_model.keras'
model = load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = request.get_json()
        state_input = data['state']
        nitrogen = float(data['nitrogen'])
        potassium = float(data['potassium'])
        phosphorus = float(data['phosphorus'])
        ph = float(data['ph'])
        temperature = float(data['temperature'])
        rainfall = float(data['rainfall'])
        humidity = float(data['humidity'])

        input_features = np.array([[nitrogen, potassium, phosphorus, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_features)

        labels = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
                  'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
                  'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
                  'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

        top_20_crop_indices = np.argsort(prediction, axis=1)[0][-20:][::-1]
        top_20_crops = [labels[index] for index in top_20_crop_indices]

        crop_images = {
            'apple': 'crop_images/apple.jpg', 'banana': 'crop_images/banana.jpg',
            'blackgram': 'crop_images/blackgram.jpg', 'chickpea': 'crop_images/chickpea.jpg',
            'coconut': 'crop_images/coconut.jpg', 'coffee': 'crop_images/coffee.jpg',
            'cotton': 'crop_images/cotton.jpg', 'grapes': 'crop_images/grapes.jpg', 'jute': 'crop_images/jute.jpg',
            'kidneybeans': 'crop_images/kidneybeans.jpg', 'lentil': 'crop_images/lentil.jpg',
            'maize': 'crop_images/maize.jpg', 'mango': 'crop_images/mango.jpg',
            'mothbeans': 'crop_images/mothbeans.jpg', 'mungbean': 'crop_images/mungbean.jpg',
            'muskmelon': 'crop_images/muskmelon.jpg',
            'orange': 'crop_images/orange.jpg', 'papaya': 'crop_images/papaya.jpg',
            'pigeonpeas': 'crop_images/pigeonpeas.jpg', 'pomegranate': 'crop_images/pomegranate.jpg',
            'rice': 'crop_images/rice.jpg', 'watermelon': 'crop_images/watermelon.jpg'
        }

        states_dict = {
            "Andhra Pradesh": ["Maize", "Pigeonpeas", "Sugarcane", "Cotton", "Banana", "Mango", "Groundnut"],
            "Arunachal Pradesh": ["Rice", "Maize", "Millet", "Ginger", "Chillies", "Oilseeds", "Orange"],
            "Assam": ["Tea", "Rice", "Maize", "Jute", "Pulses", "Oilseeds", "Sugarcane"],
            "Bihar": ["Rice", "Wheat", "Maize", "Pulses", "Oilseeds", "Sugarcane", "Jute"],
            "Chhattisgarh": ["Rice", "Maize", "Pulses", "Oilseeds", "Sugarcane", "Jute", "Tobacco"],
            "Goa": ["Rice", "Coconut", "Cashew nuts", "Arecanut"],
            "Gujarat": ["Cotton", "Groundnut", "Castor", "Bajra", "Tur", "Green gram", "Sesamum", "Paddy", "Maize","Sugarcane"],
            "Haryana": ["Wheat", "Rice", "Sugarcane", "Barley", "Gram", "Sunflower", "Rapeseed", "Mustard", "Cotton"],
            "Himachal Pradesh": ["Maize", "Wheat", "Barley", "Rice", "Apple", "Citrus fruits", "Stone fruits","Tobacco"],
            "Jharkhand": ["Rice", "Maize", "Pulses", "Oilseeds", "Sugarcane"],
            "Karnataka": ["Rice", "Maize", "Pulses", "Oilseeds", "Sugarcane", "Coffee", "Rubber", "Tea", "Cashews","Cardamom", "Chillies"],
            "Kerala": ["Coconut", "Rubber", "Coffee", "Pepper", "Cashewnuts", "Ginger", "Turmeric", "Tea", "Cardamom","Cinnamon"],
            "Madhya Pradesh": ["Wheat", "Rice", "Gram", "Maize", "Soyabean", "Pulses", "Oilseeds", "Cotton"],
            "Maharashtra": ["Rice", "Jowar", "Bajra", "Maize", "Wheat", "Pulses", "Oilseeds", "Sugarcane", "Cotton","Grapes"],
            "Manipur": ["Rice", "Maize", "Pulses", "Fruits", "Vegetables", "Spices", "Orange"],
            "Meghalaya": ["Rice", "Maize", "Pulses", "Oilseeds", "Fruits", "Vegetables", "Spices"],
            "Mizoram": ["Rice", "Maize", "Pulses", "Oilseeds", "Fruits", "Vegetables", "Spices"],
            "Nagaland": ["Rice", "Maize", "Pulses", "Oilseeds", "Fruits", "Vegetables", "Spices"],
            "Odisha": ["Rice", "Pulses", "Oilseeds", "Sugarcane", "Jute", "Cotton", "Tobacco", "Mango", "Papaya"],
            "Punjab": ["Wheat", "Rice", "Maize", "Barley", "Gram", "Mustard", "Sugarcane", "Cotton"],
            "Rajasthan": ["Wheat", "Barley", "Maize", "Millets", "Pulses", "Oilseeds", "Cotton", "Sugarcane", "Mango","Pomegranate"],
            "Sikkim": ["Rice", "Maize", "Barley", "Buckwheat", "Potatoes", "Large cardamom", "Ginger", "Fruits","Vegetables", "Orange"],
            "Tamil Nadu": ["Rice", "Jowar", "Bajra", "Maize", "Ragi", "Pulses", "Oilseeds", "Sugarcane", "Coconut", "Groundnut", "Cotton", "Coffee", "Papaya"],
            "Telangana": ["Rice", "Maize", "Pulses", "Oilseeds", "Sugarcane", "Chillies", "Turmeric", "Tobacco", "Cotton", "Mango", "Pomegranate"],
            "Tripura": ["Rice", "Maize", "Pulses", "Oilseeds", "Fruits", "Vegetables", "Rubber"],
            "Uttar Pradesh": ["Wheat", "Rice", "Sugarcane", "Barley", "Gram", "Pulses", "Oilseeds", "Cotton", "Mango", "Papaya"],
            "Uttarakhand": ["Rice", "Maize", "Wheat", "Barley", "Millets", "Pulses", "Oilseeds", "Fruits", "Vegetables"],
            "West Bengal": ["Rice", "Jute", "Sugarcane", "Wheat", "Maize", "Pulses", "Oilseeds", "Fruits", "Vegetables", "Spices", "Mango", "Orange", "Banana", "Papaya"]
        }

        crop_descriptions = {
            'rice': 'Rice is a staple food for over half of the world\'s population, particularly in Asia.',
            'coffee': 'Coffee is one of the most popular beverages in the world, known for its stimulating effects.',
            'jute': 'Jute is a long, soft, shiny bast fiber that can be spun into coarse, strong threads.',
            'pigeonpeas': 'Pigeonpeas are a drought-resistant legume crop widely grown in Asia, Africa, and the Americas.',
            'mango': 'Mango is a tropical fruit known for its sweet and juicy taste.',
            'coconut': 'Coconut is a versatile fruit used in various culinary dishes and products.',
            'maize': 'Maize, also known as corn, is a cereal grain first domesticated by indigenous peoples in southern Mexico.',
            'papaya': 'Papaya is a tropical fruit rich in vitamins, minerals, and antioxidants.',
            'chickpea': 'Chickpea, also known as garbanzo bean, is a legume widely consumed in various cuisines.',
            'apple': 'Apple is a popular fruit known for its crisp texture and sweet taste.',
            'kidneybeans': 'Kidney beans are a variety of common bean used in a variety of traditional dishes.',
            'banana': 'Banana is a nutritious fruit that is consumed worldwide.',
            'pomegranate': 'Pomegranate is a fruit with a rich history and cultural significance.',
            'orange': 'Orange is a citrus fruit known for its refreshing flavor and high vitamin C content.',
            'mothbeans': 'Moth beans are a legume crop grown primarily in India and other parts of Asia.',
            'grapes': 'Grapes are a type of fruit that grow in clusters and come in various colors.',
            'lentil': 'Lentils are edible legumes known for their lens-shaped seeds.',
            'muskmelon': 'Muskmelon, also known as cantaloupe, is a sweet and juicy fruit.',
            'watermelon': 'Watermelon is a refreshing fruit that is consumed as a hydrating summer treat.',
            'cotton': 'Cotton is a soft, fluffy staple fiber that grows in a boll, or protective case, around the seeds of the cotton plants.'
        }

        crop_list = []
        if state_input in states_dict:
            state_crops = [crop.lower() for crop in states_dict[state_input]]
            for crop in top_20_crops:
                if crop in state_crops:
                    crop_list.append({'name': crop, 'image': crop_images.get(crop, 'default.jpg'),
                                      'description': crop_descriptions.get(crop, '')})

        session['recommendedCrops'] = crop_list

        return jsonify({'redirect': url_for('results')})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/results')
def results():
    crops = session.get('recommendedCrops', [])
    print(crops)
    return render_template('result.html', crops=crops)

@app.route('/yield')
def yield_page():
    crop_list = session.get('recommendedCrops', [])
    print(crop_list)
    return render_template('yield.html', crop_list=crop_list)

with open('/Users/harsha/Documents/PCL/Code/Website/random_forest_model.pkl', 'rb') as model_file:
    model1 = pickle.load(model_file)

def predict_production(crop, season, area, ferti, pesti, rain):
    try:
        # Strip extra spaces from crop and season
        crop = crop.strip()
        season = season.strip()


        crop_mapping = {'Arecanut': 0, 'Arhar/Tur': 1, 'Bajra': 2, 'Banana': 3, 'Barley': 4, 'Black pepper': 5, 'Cardamom': 6, 'Cashewnut': 7,
                        'Castor seed': 8, 'Coconut ': 9, 'Coriander': 10, 'Cotton(lint)': 11, 'Cowpea(Lobia)': 12, 'Dry chillies': 13, 'Garlic': 14,
                        'Ginger': 15, 'Gram': 16, 'Groundnut': 17, 'Guar seed': 18, 'Horse-gram': 19, 'Jowar': 20, 'Jute': 21, 'Khesari': 22,
                        'Linseed': 23, 'Maize': 24, 'Masoor': 25, 'Mesta': 26, 'Moong(Green Gram)': 27, 'Moth': 28, 'Niger seed': 29,
                        'Oilseeds total': 30, 'Onion': 31, 'Other  Rabi pulses': 32, 'Other Cereals': 33, 'Other Kharif pulses': 34,
                        'Other Summer Pulses': 35, 'Peas & beans (Pulses)': 36, 'Potato': 37, 'Ragi': 38, 'Rapeseed &Mustard': 39, 'Rice': 40,
                        'Safflower': 41, 'Sannhamp': 42, 'Sesamum': 43, 'Small millets': 44, 'Soyabean': 45, 'Sugarcane': 46, 'Sunflower': 47,
                        'Sweet potato': 48, 'Tapioca': 49, 'Tobacco': 50, 'Turmeric': 51, 'Urad': 52, 'Wheat': 53, 'other oilseeds': 54}

        season_mapping = {'Autumn': 0, 'Kharif': 1, 'Rabi': 2, 'Summer': 3, 'Whole Year': 4, 'Winter': 5}

        # Map crop and season to label encoded values
        crop_label = crop_mapping.get(crop, 0)  # Default to 0 if not found
        season_label = season_mapping.get(season, 0)  # Default to 0 if not found

        # Combine label encoded values with other numerical inputs
        input_features = np.array([[crop_label, season_label, area, rain, ferti, pesti]])

        # Predict production using the random forest model
        production_prediction = model1.predict(input_features)

        return production_prediction[0]  # Assuming single value prediction
    except Exception as e:
        print("Error in production prediction:", e)
        return None


if __name__ == '__main__':
    app.run(debug=True)
