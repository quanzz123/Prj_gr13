# Báo cáo phân tích Solar Power Generation

## 1. Giới thiệu dữ liệu

Bộ dữ liệu gồm dữ liệu phát điện và dữ liệu thời tiết của 2 nhà máy điện mặt trời.

**Dữ liệu phát điện** gồm các cột: `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`. Trong đó `SOURCE_KEY` là mã inverter, nên mỗi thời điểm có nhiều dòng tương ứng với nhiều inverter.

**Dữ liệu thời tiết** gồm các cột: `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`. Mỗi nhà máy có một cảm biến thời tiết theo từng mốc thời gian.

Mục tiêu của bài là dự báo `AC_POWER_TOTAL`, tức tổng công suất AC của toàn nhà máy tại từng thời điểm 15 phút.

## 2. Thống kê ban đầu

### Dữ liệu phát điện

| Nhà máy | Số dòng | Số timestamp | Số inverter | AC trung bình | DC trung bình | Tỷ lệ AC = 0 |
|---|---:|---:|---:|---:|---:|---:|
| Plant 1 | 68,778 | 3,158 | 22 | 307.80 | 3,147.43 | 46.46% |
| Plant 2 | 67,698 | 3,259 | 22 | 241.28 | 246.70 | 52.68% |

### Dữ liệu thời tiết

| Nhà máy | Số dòng | Nhiệt độ môi trường TB | Nhiệt độ module TB | Irradiation TB | Irradiation max |
|---|---:|---:|---:|---:|---:|
| Plant 1 | 3,182 | 25.53 | 31.09 | 0.2283 | 1.2217 |
| Plant 2 | 3,259 | 28.07 | 32.77 | 0.2327 | 1.0988 |

Nhận xét: tỷ lệ `AC_POWER = 0` khá cao vì dữ liệu bao gồm ban đêm. Đây là hiện tượng tự nhiên, không nên xem toàn bộ là dữ liệu lỗi.

## 3. Tiền xử lý dữ liệu

Các bước đã thực hiện:

1. Chuyển `DATE_TIME` về kiểu datetime. Plant 1 dùng định dạng ngày `dd-mm-yyyy HH:MM`, Plant 2 có thể parse trực tiếp bằng pandas.
2. Gộp dữ liệu phát điện theo `PLANT_ID`, `PLANT_NO`, `DATE_TIME`.
3. Tạo các biến tổng: `DC_POWER_TOTAL`, `AC_POWER_TOTAL`, `DAILY_YIELD_TOTAL`, `TOTAL_YIELD_TOTAL`.
4. Tạo thêm `ACTIVE_SOURCE_COUNT`, `ZERO_AC_COUNT`, `ZERO_AC_RATIO` để mô tả số inverter hoạt động và tỷ lệ inverter không phát AC.
5. Ghép dữ liệu phát điện với dữ liệu thời tiết theo thời gian.
6. Tạo đặc trưng thời gian: `HOUR`, `MINUTE`, `DAYOFWEEK`, `DAYOFYEAR`, `MONTH`, `HOUR_SIN`, `HOUR_COS`, `IS_DAYLIGHT`.
7. Loại dòng thiếu weather trước khi train mô hình.

Sau khi gộp:

| Nhà máy | Số dòng sau gộp | Dòng thiếu weather | Thời gian bắt đầu | Thời gian kết thúc | AC_POWER_TOTAL TB |
|---|---:|---:|---|---|---:|
| Plant 1 | 3,158 | 1 | 2020-05-15 00:00:00 | 2020-06-17 23:45:00 | 6,703.63 |
| Plant 2 | 3,259 | 0 | 2020-05-15 00:00:00 | 2020-06-17 23:45:00 | 5,011.97 |

Nhận xét: dữ liệu sau gộp phù hợp cho mô hình dự báo theo timestamp vì mỗi dòng tương ứng với một thời điểm của một nhà máy.

## 4. Mô hình học máy

Mô hình sử dụng là `RandomForestRegressor`.

Lý do chọn mô hình:

- Phù hợp dữ liệu dạng bảng.
- Xử lý được quan hệ phi tuyến giữa bức xạ, nhiệt độ, thời gian và công suất.
- Ít yêu cầu chuẩn hóa dữ liệu.
- Có thể xem mức độ quan trọng của đặc trưng.

Đặc trưng đầu vào:

`PLANT_NO`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`, `HOUR`, `MINUTE`, `DAYOFWEEK`, `DAYOFYEAR`, `MONTH`, `HOUR_SIN`, `HOUR_COS`, `IS_DAYLIGHT`.

Biến mục tiêu:

`AC_POWER_TOTAL`.

Cách chia dữ liệu:

- Sắp xếp theo `DATE_TIME`.
- 80% đầu làm train set.
- 20% cuối làm test set.

Nhận xét: do dữ liệu có tính chuỗi thời gian, chia theo thời gian hợp lý hơn chia ngẫu nhiên.

## 5. Đánh giá kết quả

| Phạm vi | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Overall | 501.15 | 1,397.34 | 0.9588 |
| Plant 1 | 257.74 | 531.90 | 0.9955 |
| Plant 2 | 743.81 | 1,901.84 | 0.8818 |

Nhận xét:

- Mô hình đạt R2 tổng thể khoảng 0.96, cho thấy khả năng dự báo tốt.
- Plant 1 có kết quả rất tốt với R2 khoảng 0.9955.
- Plant 2 có sai số cao hơn và R2 thấp hơn. Điều này cho thấy Plant 2 khó dự báo hơn, có thể do biến động công suất lớn hơn, dữ liệu nhiễu hơn hoặc đặc điểm vận hành khác Plant 1.

## 6. Trực quan hóa

Các biểu đồ đã xuất trong thư mục `outputs`:

- `daily_generation_comparison.png`: so sánh tổng AC power theo ngày giữa 2 nhà máy.
- `daily_weather_comparison.png`: so sánh irradiation và nhiệt độ module theo ngày.
- `plant_1_correlation_heatmap.png`, `plant_2_correlation_heatmap.png`: heatmap tương quan.
- `plant_1_predictions.png`, `plant_2_predictions.png`: so sánh giá trị thực tế và dự đoán.
- `feature_importance.png`: mức độ quan trọng của đặc trưng.

## 7. Nhận xét đặc trưng quan trọng

Đặc trưng quan trọng nhất là `IRRADIATION`, chiếm khoảng 92.88% importance trong mô hình Random Forest. Đây là kết quả hợp lý vì bức xạ mặt trời là yếu tố trực tiếp quyết định công suất phát điện.

Các đặc trưng tiếp theo gồm `PLANT_NO`, `DAYOFYEAR`, `DAYOFWEEK`, nhiệt độ môi trường, nhiệt độ module và đặc trưng chu kỳ giờ. Điều này cho thấy ngoài thời tiết, từng nhà máy và yếu tố thời gian cũng ảnh hưởng đến công suất.

## 8. Kết luận

Quy trình khai phá dữ liệu đã hoàn thành các bước chính: đọc dữ liệu, giải thích ý nghĩa cột, tiền xử lý, ghép dữ liệu, tạo đặc trưng, huấn luyện Random Forest, đánh giá bằng MAE/RMSE/R2 và trực quan hóa kết quả.

Kết quả cho thấy Random Forest là mô hình phù hợp cho bài toán dự báo công suất điện mặt trời dạng bảng. Tuy nhiên, để cải thiện Plant 2 có thể thử train riêng từng nhà máy, thêm đặc trưng trễ theo chuỗi thời gian hoặc thử các mô hình như Gradient Boosting, XGBoost hoặc LSTM.
