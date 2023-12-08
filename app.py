from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Memuat model saat aplikasi dimulai
model = load_model('model.h5')

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
    
    input_array = [
        gender, input_data["age"], major,
        input_data["year"], input_data["cgpa"], marriage,
        anxiety, panic, treatment
    ]
    # Contoh penggunaan model untuk membuat prediksi
    # Pastikan untuk mengubah kode ini sesuai dengan model Anda
    prediction_result = model.predict([input_array])
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
                    "prediction": depression_prediction,
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

if __name__ == "__main__":
    app.run()