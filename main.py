from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)


def run_training_and_forecast():
    # ======= GOOGLE SHEET =========
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    google_key = os.environ.get("GOOGLE_SERVICE_KEY")
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(google_key), scope)
    client = gspread.authorize(creds)

    sheet_url = "https://docs.google.com/spreadsheets/d/19qBwHPrIes6PeGAyIzMORPVB-7utQpaZG7RHrdRfoNI/edit#gid=0"
    sheet = client.open_by_url(sheet_url)
    worksheet = sheet.worksheet("DATA")
    data = pd.DataFrame(worksheet.get_all_records())

    # ======= TIỀN XỬ LÝ =========
    data['timestamp'] = pd.to_datetime(data['NGÀY'] + ' ' + data['GIỜ'], format='%d/%m/%Y %H:%M:%S')
    data = data.sort_values('timestamp')
    data.rename(columns={
        'temperature': 'temp',
        'humidity': 'humid',
        'soil_moisture': 'soil',
        'wind': 'wind',
        'rain': 'rain'
    }, inplace=True)

    # ======= CHECK TIMESTAMP =========
    saved_timestamp = None
    if os.path.exists("last_timestamp.json"):
        with open("last_timestamp.json", "r") as f:
            saved_timestamp = pd.to_datetime(json.load(f)["last_timestamp"])
    latest_timestamp = data['timestamp'].iloc[-1]

    # ======= CHUẨN HÓA =========
    features = ['temp', 'humid', 'soil', 'wind', 'rain']
    dataset = data[features].copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)

    model_path = "gru_weather_model.h5"
    window_size = 6

    if not os.path.exists(model_path):
        print("⚠️ Chưa có mô hình, sẽ tạo mới.")
        model = Sequential()
        model.add(GRU(units=64, return_sequences=False, input_shape=(window_size, len(features))))
        model.add(Dense(5))
        model.compile(optimizer='adam', loss=MeanSquaredError())
        model.save(model_path)
    else:
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss=MeanSquaredError())

    # ======= DỰ BÁO =========
    n_steps = 25
    forecast = []
    current_seq = scaled_data[-window_size:].copy()

    for _ in range(n_steps):
        x_input = current_seq.reshape(1, window_size, len(features))
        y_pred = model.predict(x_input, verbose=0)
        forecast.append(y_pred[0])
        current_seq = np.vstack([current_seq[1:], y_pred])

    forecast_original = scaler.inverse_transform(np.array(forecast))
    forecast_df = pd.DataFrame(forecast_original, columns=features)
    forecast_df = forecast_df.clip(lower=0).round(2)

    base_time_tomorrow = (datetime.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    forecast_df.insert(0, "time", [(base_time_tomorrow + timedelta(hours=i)).strftime("%d/%m/%Y %H:%M") for i in range(n_steps)])

    forecast_df.to_json("latest_prediction.json", orient="records", indent=2)
    print("📤 Đã lưu latest_prediction.json")

    # ======= FIREBASE =========
    if not firebase_admin._apps:
        firebase_key = os.environ.get("FIREBASE_SERVICE_KEY")
        if not firebase_key:
            raise ValueError("FIREBASE_SERVICE_KEY không tồn tại trong biến môi trường!")
        cred = credentials.Certificate(json.loads(firebase_key))
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://smart-farm-6e42d-default-rtdb.firebaseio.com/'
        })

    ref = db.reference("forecast/tomorrow")
    ref.set(forecast_df.to_dict(orient="records"))
    print("🔥 Đã đẩy dữ liệu lên Firebase")

    # ======= HUẤN LUYỆN =========
    if saved_timestamp is not None and latest_timestamp <= saved_timestamp:
        print("🟡 Không có dữ liệu mới.")
        return

    print("🟢 Có dữ liệu mới. Đang huấn luyện...")
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size])
    X = np.array(X)
    y = np.array(y)

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=16, callbacks=[early_stop], verbose=0)
    model.save(model_path)

    with open("last_timestamp.json", "w") as f:
        json.dump({"last_timestamp": str(latest_timestamp)}, f)
    print("✅ Đã huấn luyện xong.")


@app.route('/weather/predict', methods=['GET'])
def get_prediction():
    try:
        if not os.path.exists("latest_prediction.json"):
            return jsonify({"error": "❌ File chưa tồn tại"}), 404
        df = pd.read_json("latest_prediction.json", orient="records")
        df = df[["time", "temp", "humid", "soil", "wind", "rain"]]
        return jsonify({"forecast": df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======= KHỞI TẠO SCHEDULER =========
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_training_and_forecast, 'interval', minutes=10)
    scheduler.start()
    print("🌀 Scheduler đang chạy mỗi 10 phút...")


if __name__ == '__main__':
    start_scheduler()
    run_training_and_forecast()  # chạy ngay lần đầu tiên
    app.run(host='0.0.0.0', port=8080)
