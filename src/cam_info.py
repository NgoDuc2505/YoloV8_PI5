from ultralytics import YOLO
import cv2
import logging




# Resolution: 640x480
# Width: 640
# Height: 480
# Channels: 3

def print_webcam_info(frame):
    height, width = frame.shape
    if(len(frame.shape) == 3):
        height, width, channels = frame.shape
        resolution = f"{width}x{height}"
        print(f"Resolution: {resolution}")
        print(f"Width: {width}")
        print(f"Height: {height}")
        print(f"Channels: {channels}")
    else:
        print(f"Resolution: {width}x{height}")
        print(f"Width: {width}")
        print(f"Height: {height}")
        print(f"Channels: {1}")


def webcam_stream_with_YOLO(width=640, height=480, gray=False, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    model = YOLO('models/yolov8n.pt')
    printed = False
    # Cấu hình độ phân giải
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError("Không thể mở webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ webcam.")
            break
        results = model(frame)
        # Resize để đảm bảo đúng kích thước
        frame = cv2.resize(frame, (width, height))

        # Chuyển sang ảnh xám nếu cần
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not printed:
            print_webcam_info(frame)
            printed = True

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
        cv2.imshow("Webcam Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()