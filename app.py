import traceback
import numpy as np

from flask import Flask, jsonify, request
from keras.models import load_model
from PIL import Image
from io import BytesIO
from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

# Memuat model saat aplikasi dimulai
model_depression = load_model('model.h5')

# Load the TFLite model
interpreter = Interpreter(model_path="model-float-tl.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def convert_gender(gender_string):
    if gender_string.lower() == "female":
        return 0
    elif gender_string.lower() == "male":
        return 1
    else:
        # Nilai default atau error handling jika tidak sesuai
        return -1

def convert_major(major_string):
    if major_string.lower() == "ekonomi":
        return 10
    elif major_string.lower() == "hukum":
        return 15
    elif major_string.lower() == "engineering":
        return 17
    elif major_string.lower() == "law":
        return 15
    elif major_string.lower() == "sistem informasi":
        return 20
    elif major_string.lower() == "teknik informatika":
        return 31
    elif major_string.lower() == "sipil":
        return 32
    elif major_string.lower() == "kedokteran":
        return 35
    else:
        return 9
    
def convert_marriage(marriage_string):
    if marriage_string.lower() == "single":
        return 0
    elif marriage_string.lower() == "menikah":
        return 1
    else:
        # Nilai default atau error handling jika tidak sesuai
        return -1
    
def convert_anxiety(anxiety_string):
    if anxiety_string.lower() == "no":
        return 0
    elif anxiety_string.lower() == "yes":
        return 1
    else:
        # Nilai default atau error handling jika tidak sesuai
        return -1
    
def convert_panic(panic_string):
    if panic_string.lower() == "no":
        return 0
    elif panic_string.lower() == "yes":
        return 1
    else:
        # Nilai default atau error handling jika tidak sesuai
        return -1
    
def convert_treatment(treatment_string):
    if treatment_string.lower() == "no":
        return 0
    elif treatment_string.lower() == "yes":
        return 1
    else:
        # Nilai default atau error handling jika tidak sesuai
        return -1

# Fungsi prediksi menggunakan model ML yang dimuat
def predict(input_data):
    
    # Konversi nilai "gender" dari string ke integer sebelum dimasukkan ke input_array
    gender = convert_gender(input_data["gender"])
    major = convert_major(input_data["major"])
    marriage = convert_marriage(input_data["marriage"])
    anxiety = convert_anxiety(input_data["anxiety"])
    panic = convert_panic(input_data["panic"])
    treatment = convert_treatment(input_data["treatment"])
    
    cgpa = float(input_data["cgpa"])
    if cgpa >= 4:
        cgpa_score = 0
    elif cgpa >= 3:
        cgpa_score = 1
    elif cgpa >= 2:
        cgpa_score = 2
    elif cgpa >= 1:
        cgpa_score = 3
    else:
        cgpa_score = 4
    
    input_array = [
        gender, input_data["age"], major,
        input_data["year"], cgpa_score, marriage,
        anxiety, panic, treatment
    ]
    # Contoh penggunaan model untuk membuat prediksi
    # Pastikan untuk mengubah kode ini sesuai dengan model Anda
    prediction_result = model_depression.predict([input_array])
    return prediction_result

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Succes Fetching the API",
        },
        "data": None
    }), 200
    
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        data = request.get_json()
        required_fields = ["gender", "age", "major", "year", "cgpa", "marriage", "anxiety", "panic", "treatment"]
        if all(field in data for field in required_fields):
            result = predict(data)  # Memanggil fungsi prediksi dengan data yang lengkap
            depression_prediction = result.item()  # Misalnya, hasil prediksi Depression berada di index pertama dari hasil prediksi
            
            prediction_percentage = depression_prediction * 100
            prediction_result = int(prediction_percentage)
            
            if depression_prediction > 0.5:
                depression_result = "Iya"
            else:
                depression_result = "Tidak"
                
            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success",
                },
                "data": {
                    "prediction": prediction_result,
                    "depression": depression_result
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": f"Bad Request: Required fields are missing. Required fields: {', '.join(required_fields)}"
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "massage": "Method not Allowed"
            },
            "data": None
        }), 405
        
@app.route("/expression", methods=["GET", "POST"])
def expression():
    try:
        file = request.files['file']
        
        # Ensure the uploaded file is an image
        if file.content_type not in ["image/jpeg", "image/png"]:
            return jsonify({"error": "File is not an image"}), 400
        
        # Preprocess the image
        img = Image.open(BytesIO(file.read()))
        img = img.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Set the image as input to the model
        interpreter.set_tensor(input_details[0]['index'], [img_array])
        interpreter.invoke()
        
        # Get the model output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Map the output to emotions
        emotions = {
            0: "marah",
            1: "penghinaan",
            2: "jijik",
            3: "takut",
            4: "senang",
            5: "sedih",
            6: "kejutan"
        }
        
        # Convert model output to emotions label
        predicted_class = np.argmax(output_data)
        emotion_label = emotions.get(predicted_class, "Tidak dikenali")
        
        return jsonify({"emotion": emotion_label}), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run()