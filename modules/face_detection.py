import torch
import cv2
from ultralytics import YOLO


class FaceDetector:
    """
    Face detection using YOLO model (e.g. YOLOv8 face detection).
    Runs on GPU (MPS/CUDA) or CPU.
    """
    def __init__(self, device=None, model_path="yolov8n-face.pt", resize_factor=None, skip_frames=None):
        # Select device: MPS > CUDA > CPU
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device = device
        # Load YOLO face detection model
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, frame):
        """
        Detect faces in a BGR frame using YOLO.
        Returns lists of bounding boxes and confidences.
        """
        # Run inference (frame in BGR)
        results = self.model(frame)
        boxes, probs = [], []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
                probs.append(conf)
        return boxes, probs

    def crop_faces(self, frame, boxes):
        """
        Crop detected face regions from BGR frame.
        boxes: list of (x1, y1, x2, y2)
        Returns list of cropped face images (BGR).
        """
        crops = []
        for (x1, y1, x2, y2) in boxes:
            crop = frame[y1:y2, x1:x2]
            crops.append(crop)
        return crops
