Xây dựng một AI pipeline có khả năng:
- Nhận diện vị trí các cầu thủ (và quả bóng) từ từng khung hình video.
- Phân loại số áo cầu thủ dựa trên ảnh crop được từ model phát hiện.

Pipeline bao gồm hai mô hình chính:
1. Object Detection: Phát hiện vị trí các cầu thủ.
2. Image Classification: Phân loại số áo cầu thủ từ ảnh crop.

1. Object Detection - Phát hiện cầu thủ
  Nhiệm vụ:
Phát hiện vị trí các cầu thủ (và quả bóng) từ từng khung hình.
Sử dụng các model YOLO từ Ultralytics:  YOLOv11.
  Tài nguyên:
Repo: ultralytics/ultralytics
Hướng dẫn: docs.ultralytics.com/tasks/detect

  Các bước triển khai:
1. Chuẩn bị dữ liệu:
- Tách video thành từng frame.
- Ánh xạ đúng file .json annotation cho từng frame.
- Chuyển annotation về định dạng YOLO.
Tài liệu: Data format YOLO

2. Huấn luyện mô hình:
- Qua Python, CLI, hoặc Ultralytics Hub.
- Điều chỉnh tham số huấn luyện phù hợp (imgsz, epochs, batch, lr, ...).
Kết quả:
Mô hình đầu ra bounding boxes của cầu thủ (và bóng) trong từng khung hình video.

2. Image Classification - Phân loại số áo
  Nhiệm vụ:
- Nhận ảnh của từng cầu thủ được crop từ YOLO model.
- Phân loại số áo dựa vào ảnh đầu vào.
Định nghĩa class:
- Class 0: Không thấy rõ số áo.
- Class 1-10: Các số từ 1 đến 10.
- Class 11: Từ số 11 trở lên.

Lưu ý quan trọng: Các model sử dụng được train qua kaggle để sử dụng 
