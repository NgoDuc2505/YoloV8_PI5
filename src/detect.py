import cv2
from ultralytics import YOLO



def detect_objects():
    # Tải mô hình YOLOv8n đã được huấn luyện sẵn
    model = YOLO('yolov8n.pt')

    # Mở camera (0 cho camera mặc định)
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Thực hiện nhận diện đối tượng
        results = model(frame)

        # Lọc kết quả để chỉ hiển thị người và xe
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                if cls_name in ['person', 'car', 'bus', 'truck', 'motorcycle']:
                    # Vẽ khung bao quanh đối tượng
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, cls_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Hiển thị kết quả
        cv2.imshow('YOLOv8n Detection', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
