# Solar Power Generation Data Analysis

Du an phan tich du lieu phat dien mat troi cua hai nha may `Plant 1` va `Plant 2`.
Trong phien ban hien tai, luong lam viec chinh tap trung vao:

- Lay them du lieu thoi tiet tu Open-Meteo API.
- Kham pha va truc quan hoa cau truc du lieu.
- Tien xu li va ghep du lieu generation voi weather sensor o cap plant/timestamp.
- Chuan hoa feature va chia train/test theo thoi gian trong notebook.

Notebook EDA khong huan luyen mo hinh. Cac file/model cu trong repo duoc giu lai de tham khao.

## Cau truc du lieu

Du lieu goc co hai nhom file cho moi plant:

- `Plant_X_Generation_Data.csv`: du lieu phat dien theo tung inverter.
- `Plant_X_Weather_Sensor_Data.csv`: du lieu cam bien thoi tiet cua nha may.

Trong generation, moi timestamp co nhieu dong tu nhieu inverter. Vi vay khong merge truc tiep weather vao tung inverter neu muc tieu la mo hinh cap nha may. Luong EDA moi se tong hop generation theo:

```text
PLANT_ID, PLANT_NO, DATE_TIME
```

sau do moi ghep weather sensor va Open-Meteo theo timestamp.

## File chinh

| File | Vai tro |
|---|---|
| `weather_api_fetcher.py` | Goi Open-Meteo API va luu du lieu thoi tiet thanh CSV theo plant. |
| `EDASolarPowerGenerationData.ipynb` | Notebook chinh cho EDA, tien xu li, merge, truc quan hoa, chuan hoa va chia train/test trong bo nho. Khong train model. |
| `prepare_train_test_data.py` | Script tuy chon de chuan hoa va luu train/test ra CSV. Hien khong phai luong uu tien vi notebook da lam trong bo nho. |
| `solar_power_analysis.py` | Script cu co train Random Forest va tao bao cao/plot. Dung de tham khao hoac so sanh. |
| `featureMeteo.py` | Script thu nghiem cu de enrich weather. Khong khuyen nghi dung cho luong plant-level moi. |

## Thu muc

```text
.
+-- datasets/
|   +-- Plant_1_Generation_Data.csv
|   +-- Plant_1_Weather_Sensor_Data.csv
|   +-- Plant_2_Generation_Data.csv
|   +-- Plant_2_Weather_Sensor_Data.csv
|   +-- api_weather/
|       +-- open_meteo_weather_plant_1.csv
|       +-- open_meteo_weather_plant_2.csv
+-- outputs/
|   +-- eda_all_plants_merged.csv
|   +-- eda_plant_1_merged.csv
|   +-- eda_plant_2_merged.csv
+-- weather_api_fetcher.py
+-- EDASolarPowerGenerationData.ipynb
+-- prepare_train_test_data.py
+-- solar_power_analysis.py
```

Mot so file CSV goc cung co the nam truc tiep o thu muc root. Cac ham load du lieu se uu tien `datasets/`, sau do fallback ve root.

## Cai dat

Yeu cau Python 3.8+.

```bash
pip install pandas numpy matplotlib requests scikit-learn
```

`scikit-learn` chi can cho cac script/model cu. Notebook EDA hien tai khong can train model.

## 1. Lay du lieu thoi tiet Open-Meteo

Chay cho ca hai plant:

```bash
python weather_api_fetcher.py
```

Chay rieng tung plant:

```bash
python weather_api_fetcher.py --plant 1
python weather_api_fetcher.py --plant 2
```

Ket qua duoc luu vao:

```text
datasets/api_weather/open_meteo_weather_plant_1.csv
datasets/api_weather/open_meteo_weather_plant_2.csv
```

Script lay khoang thoi gian tu du lieu goc, goi endpoint Open-Meteo historical archive, sau do can du lieu hourly ve moc 15 phut bang forward-fill de merge voi dataset solar.

### Cac cot Open-Meteo

Tat ca cot API duoc dat prefix `OM_`:

| Cot | Y nghia |
|---|---|
| `OM_TEMPERATURE_2M` | Nhiet do khong khi o do cao 2m. |
| `OM_RELATIVE_HUMIDITY_2M` | Do am tuong doi o do cao 2m. |
| `OM_DEW_POINT_2M` | Nhiet do diem suong. |
| `OM_APPARENT_TEMPERATURE` | Nhiet do cam nhan. |
| `OM_PRESSURE_MSL` | Ap suat quy doi ve muc nuoc bien. |
| `OM_SURFACE_PRESSURE` | Ap suat tai be mat dia hinh. |
| `OM_PRECIPITATION` | Luong mua theo gio. |
| `OM_CLOUD_COVER` | Do che phu may. |
| `OM_WIND_SPEED_10M` | Toc do gio o do cao 10m. |
| `OM_WIND_DIRECTION_10M` | Huong gio o do cao 10m. |
| `OM_SHORTWAVE_RADIATION` | Buc xa song ngan toi be mat ngang. |
| `OM_DIRECT_RADIATION` | Thanh phan buc xa truc tiep. |
| `OM_DIFFUSE_RADIATION` | Thanh phan buc xa khuech tan. |
| `OM_DIRECT_NORMAL_IRRADIANCE` | Buc xa truc tiep vuong goc voi tia mat troi. |
| `OM_SUNSHINE_DURATION` | Thoi luong nang trong gio. |

Luu y: Open-Meteo la du lieu hourly, khong phai cam bien truc tiep tai nha may. Cac toa do trong `weather_api_fetcher.py` dang duoc hard-code; neu co metadata chinh thuc, cap nhat `PLANT_COORDINATES`.

## 2. Chay notebook EDA

Mo va chay:

```text
EDASolarPowerGenerationData.ipynb
```

Notebook thuc hien cac buoc:

1. Load generation va weather sensor cho Plant 1/2.
2. Parse `DATE_TIME` dung dinh dang theo tung plant.
3. Kiem tra schema, missing values, duplicate keys.
4. Truc quan hoa cau truc bang, sampling time, phan phoi feature, boxplot, inverter/source key.
5. Tong hop generation tu cap inverter ve cap plant/timestamp.
6. Merge generation aggregated voi weather sensor.
7. Merge them Open-Meteo neu file API da ton tai.
8. Tao time features:
   - `HOUR`
   - `MINUTE`
   - `DAY`
   - `DAYOFWEEK`
   - `WEEKOFYEAR`
   - `MONTH`
   - `DAYOFYEAR`
   - `HOUR_SIN`
   - `HOUR_COS`
   - `IS_DAYLIGHT`
9. Kiem tra chat luong du lieu sau merge.
10. Chuan hoa feature va chia train/test theo thoi gian trong bo nho.

Notebook tao cac bien sau de dung cho notebook train model ve sau:

```python
X_train
X_test
y_train
y_test
train_context
test_context
preprocessing_metadata
```

Mac dinh notebook khong luu train/test ra CSV. Cell cuoi co bien:

```python
SAVE_MERGED_CSV = False
```

Doi thanh `True` neu muon luu file merged EDA vao `outputs/`.

## 3. Du lieu dung cho training

Neu train model trong notebook khac, nen chay `EDASolarPowerGenerationData.ipynb` truoc, sau do dung truc tiep:

```python
X_train, X_test, y_train, y_test
```

Target mac dinh:

```text
AC_POWER_TOTAL
```

Feature mac dinh gom:

```text
AMBIENT_TEMPERATURE
MODULE_TEMPERATURE
IRRADIATION
HOUR
MINUTE
DAYOFWEEK
DAYOFYEAR
MONTH
HOUR_SIN
HOUR_COS
IS_DAYLIGHT
OM_* neu co va USE_OPEN_METEO_FEATURES = True
PLANT_NO_1
PLANT_NO_2
```

Cac cot khong nen dua vao feature mac dinh vi co nguy co leakage hoac chi la dinh danh:

```text
DC_POWER_TOTAL
DAILY_YIELD_TOTAL
TOTAL_YIELD_TOTAL
ZERO_AC_COUNT
ZERO_AC_RATIO
ACTIVE_SOURCE_COUNT
PLANT_ID
DATE_TIME
WEATHER_SENSOR_KEY
```

`DATE_TIME` chi nen dung de sort/split theo thoi gian hoac tao feature thoi gian.

## 4. Chuan hoa va chia train/test

Trong notebook:

- Chia theo thoi gian, khong random split.
- Mac dinh `TEST_SIZE = 0.2`.
- Fit median imputer tren train.
- Fit standard scaler tren train.
- Apply imputer/scaler cho ca train va test.
- One-hot encode `PLANT_NO`.
- Giu `AC_POWER_TOTAL` o scale goc.

Day la cach tranh ro ri du lieu tu test vao train.

Neu van muon xuat train/test ra CSV, co the dung script tuy chon:

```bash
python prepare_train_test_data.py
```

Hoac khong dung Open-Meteo:

```bash
python prepare_train_test_data.py --no-open-meteo
```

Script nay se luu:

```text
outputs/train_data_normalized.csv
outputs/test_data_normalized.csv
outputs/train_test_metadata.json
```

## 5. Ghi chu ve cac file cu

- `solar_power_analysis.py` la luong cu co train Random Forest, tao metrics va plots trong `outputs/`.
- `solar_power_lstm_notebook.ipynb` la notebook LSTM cu.
- `featureMeteo.py` la script enrich weather cu, merge weather vao tung inverter row. Khong phu hop neu muc tieu la plant-level model.
- `Plant_X_Enriched_With_Weather.csv` la output cu tu `featureMeteo.py`, khong nen dung lam input chinh cho plant-level training.

## 6. Hinh anh truc quan va ket qua mo hinh cu

Phan nay tong hop mot so output da co trong `outputs/`. Cac hinh va metrics nay den tu luong phan tich/model cu, khong phai tu notebook EDA moi.

### Truc quan du lieu quan trong

So sanh tong AC Power theo ngay giua hai plant:

![Daily Generation](outputs/daily_generation_comparison.png)

So sanh weather sensor theo ngay, gom irradiation va module temperature:

![Daily Weather](outputs/daily_weather_comparison.png)

Heatmap tuong quan dac trung theo tung plant:

| Plant 1 | Plant 2 |
| :---: | :---: |
| ![Plant 1 Correlation](outputs/plant_1_correlation_heatmap.png) | ![Plant 2 Correlation](outputs/plant_2_correlation_heatmap.png) |

Feature importance cua Random Forest:

![Feature Importance](outputs/feature_importance.png)

### Ket qua Random Forest cu

Nguon: `outputs/metrics.json` va `outputs/metrics_from_notebook.csv`.

| Scope | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Overall | 501.15 | 1397.34 | 0.9588 |
| Plant 1 | 257.74 | 531.90 | 0.9955 |
| Plant 2 | 743.81 | 1901.84 | 0.8818 |

Du doan vs thuc te tren tap test:

| Plant 1 | Plant 2 |
| :---: | :---: |
| ![Plant 1 Predictions](outputs/plant_1_predictions.png) | ![Plant 2 Predictions](outputs/plant_2_predictions.png) |

### Ket qua LSTM cu

Nguon: `outputs/lstm_metrics.csv`.

| Scope | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Overall | 1090.35 | 2053.79 | 0.9108 |
| Plant 1 | 1068.96 | 2024.31 | 0.9351 |
| Plant 2 | 1111.07 | 2081.96 | 0.8590 |

Luu y: cac ket qua tren chi nen xem la baseline/tham khao. Neu thay doi pipeline EDA, feature Open-Meteo, cach split, hoac cach chuan hoa, can train va danh gia lai.

## 7. Luong lam viec khuyen nghi

1. Dam bao CSV goc nam trong `datasets/` hoac root.
2. Lay Open-Meteo neu can:

   ```bash
   python weather_api_fetcher.py
   ```

3. Chay toan bo `EDASolarPowerGenerationData.ipynb`.
4. Dung `X_train`, `X_test`, `y_train`, `y_test` cho notebook/model training rieng.

## 8. Tac gia

Du an mon Khai pha du lieu - Nhom 13.
