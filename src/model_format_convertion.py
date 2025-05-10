import torch
from ultralytics import YOLO
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def convert_modelv5_to_onnx(model_path, output_path):
    # Tải mô hình YOLOv8 đã huấn luyện
    model = torch.hub.load('ultralytics/yolov5', 'v5.0', pretrained=True)
    dummy_input = torch.randn(1, 3, 640, 640)  # Tạo một tensor giả để kiểm tra
    onnx_path = "yolov8_model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"])

def convert_modelv8_to_onnx(model_path, output_path):
    model = YOLO(model_path)
    model.export(format="onnx", imgsz=640, opset=12, dynamic=True, simplify=True)

def convert_onnx_to_tf(onnx_path, tf_out_path="yolov8_model_tf"):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_out_path)

def convert_tf_to_tflite(tflite_out_path="yolov8_model_tflite.tflite"):
    model = tf.saved_model.load("yolov8_model_tf")

    converter = tf.lite.TFLiteConverter.from_saved_model("yolov8_model_tf")
    tflite_model = converter.convert()

    with open(f"{tflite_out_path}", "wb") as f:
        f.write(tflite_model)

def full_flow_convert_model(model_path_pt, output_path):
    out_onnx_path = f"{output_path}/yolov8_model.onnx"
    out_tf_path = f"{output_path}/yolov8_model_tf"
    out_tflite_path = f"{output_path}/yolov8_model.tflite"
    convert_modelv8_to_onnx(model_path_pt, out_onnx_path)
    convert_onnx_to_tf(out_onnx_path, out_tf_path)
    convert_tf_to_tflite(out_tflite_path)

# full_flow_convert_model("models/yolov8n.pt", "converted_models")
# convert_modelv8_to_onnx("models/yolov8n.pt", "converted_models/yolov8n.onnx")
convert_onnx_to_tf("converted_models/yolov8n.onnx", "converted_models/yolov8n_tf")