import onnxruntime as ort
import cv2
import numpy as np

class_names = ['person', 'car', 'bus', 'truck', 'motorcycle']

def detect_onnx(model_path):
    # Tải mô hình ONNX
    ort_session = ort.InferenceSession(model_path)

    # Mở webcam (cổng USB)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Tiền xử lý frame: thay đổi kích thước và chuẩn hóa
        input_frame = cv2.resize(frame, (640, 640))  # Thay đổi kích thước nếu cần thiết
        input_frame = np.transpose(input_frame, (2, 0, 1))  # Chuyển đổi thành (C, H, W)
        input_frame = np.expand_dims(input_frame, axis=0)  # Thêm batch dimension
        input_frame = input_frame.astype(np.float32)

        input_name = ort_session.get_inputs()[0].name

        # Chạy mô hình
        # outputs = ort_session.run(None, {"input": input_frame})
        outputs = ort_session.run(None, {input_name: input_frame})
        
        
        # Xử lý kết quả
        # print(outputs)  # Hiển thị kết quả đầu ra
        output_data = outputs[0]  # Dự đoán đầu tiên
        for detection in output_data:
            x1, y1, x2, y2, confidence, class_id = detection[:6]  # Lấy tọa độ và confidence
            
            # Nếu confidence là mảng, lấy phần tử đầu tiên
            if isinstance(confidence, np.ndarray):
                confidence = confidence[0]

            # Kiểm tra nếu confidence > 0.5
            if confidence > 0.3 and int(class_id) < len(class_names):
                class_name = class_names[int(class_id)]
                print(f"Detected object with class ID {class_id} at [{x1}, {y1}, {x2}, {y2}] with confidence {confidence}")

                # Vẽ bounding box lên frame
                label = f"Class {class_name} ({confidence:.2f})"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # Hiển thị frame
        cv2.imshow("Webcam Feed", frame)

        # Dừng khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def test_onnx_model(model_path):
    import onnxruntime as ort

    ort_session = ort.InferenceSession(model_path)

    # Kiểm tra tên đầu vào
    input_names = [input.name for input in ort_session.get_inputs()]
    print("Input names:", input_names)

if __name__ == "__main__":
    detect_onnx("models/yolov8n.onnx")

