from detect import detect_objects
from cam_info import webcam_stream_with_YOLO

if __name__ == "__main__":
    detect_objects()
    # webcam_stream_with_YOLO(width=480, height=480, gray=True, camera_index=1)
