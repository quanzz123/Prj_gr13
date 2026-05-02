import pandas as pd
import requests
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ====================== CẤU HÌNH ======================
#PLANT = 1  
PLANT = 2                                    # Đổi thành 2 nếu làm Plant 2
LAT = 14.815 if PLANT == 1 else 19.997
LON = 78.287 if PLANT == 1 else 73.790

# Đường dẫn file (thay theo máy bạn)
gen_path = f'Plant_{PLANT}_Generation_Data.csv'
sensor_path = f'Plant_{PLANT}_Weather_Sensor_Data.csv'

# ====================== 1. LOAD DATASET ======================
gen = pd.read_csv(gen_path, parse_dates=['DATE_TIME'])
sensor = pd.read_csv(sensor_path, parse_dates=['DATE_TIME'])

# Merge generation + sensor theo DATE_TIME
df = pd.merge(gen, sensor, on=['DATE_TIME', 'PLANT_ID'], how='left')

# Lấy khoảng thời gian chính xác của dataset
start_date = df['DATE_TIME'].min().strftime('%Y-%m-%d')
end_date   = df['DATE_TIME'].max().strftime('%Y-%m-%d')
print(f"Thời gian dataset: {start_date} → {end_date}")

# ====================== 2. GỌI API OPEN-METEO ======================
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "relative_humidity_2m,pressure_msl,precipitation,cloud_cover,wind_speed_10m,wind_direction_10m",
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params)
data = response.json()

# Chuyển thành DataFrame hourly
hourly = pd.DataFrame({
    "time": pd.to_datetime(data["hourly"]["time"]),
    "humidity": data["hourly"]["relative_humidity_2m"],
    "pressure": data["hourly"]["pressure_msl"],
    "precipitation": data["hourly"]["precipitation"],
    "cloud_cover": data["hourly"]["cloud_cover"],
    "wind_speed": data["hourly"]["wind_speed_10m"],
    "wind_direction": data["hourly"]["wind_direction_10m"]
})

print("✅ Đã lấy dữ liệu hourly từ Open-Meteo")

# ====================== 3. RESAMPLE 15 PHÚT ======================
# Tạo index 15 phút từ min → max của dataset
full_index = pd.date_range(
    start=df['DATE_TIME'].min(),
    end=df['DATE_TIME'].max(),
    freq='15T'
)

# Resample hourly → 15 phút (dùng forward fill)
hourly = hourly.set_index('time').reindex(full_index, method='ffill').reset_index()
hourly = hourly.rename(columns={'index': 'DATE_TIME'})

# ====================== 4. MERGE VÀO DATASET GỐC ======================
final_df = pd.merge(df, hourly, on='DATE_TIME', how='left')

# Lưu file mới
final_df.to_csv(f'Plant_{PLANT}_Enriched_With_Weather.csv', index=False)
print(f"✅ Đã lưu file Plant_{PLANT}_Enriched_With_Weather.csv")
print(f"Số features mới thêm: 6 (humidity, pressure, precipitation, cloud_cover, wind_speed, wind_direction)")