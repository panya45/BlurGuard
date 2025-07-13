import torch
from facenet_pytorch import MTCNN
import cv2


class FaceDetector:
    """
    Face detection using MTCNN from facenet-pytorch.
    Runs on MPS/GPU if available, otherwise CPU.
    """
    def __init__(self, device=None, keep_all=True, thresholds=[0.6, 0.7, 0.7], resize_factor=0.25, skip_frames=2):
        # Select device: MPS > CUDA > CPU
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device = device
        # Initialize MTCNN
        self.mtcnn = MTCNN(
            keep_all=keep_all,
            device=self.device,
            thresholds=thresholds
        )
        # Downscale input frames for faster detection (e.g. 0.25 => quarter size)
        self.resize_factor = resize_factor
        # Skip detection on intermediate frames to improve FPS
        self.skip_frames = skip_frames
        self._frame_counter = 0
        self._last_boxes = []
        self._last_probs = []

    @staticmethod
    def _is_mps_pool_error(err):
        return 'Adaptive pool MPS' in str(err)

    def detect(self, frame):
        """
        Detect faces in a BGR frame, fallback to CPU if MPS pooling error.
        Returns:
          - boxes: list of [x1, y1, x2, y2]
          - probs: list of face probabilities
        """
        # Convert BGR to RGB and optionally downsample
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.resize_factor != 1.0:
            h, w = rgb_full.shape[:2]
            small = cv2.resize(rgb_full, (int(w * self.resize_factor), int(h * self.resize_factor)))
        else:
            small = rgb_full
        # Skip detection on frames except every nth, only if previous detection exists
        self._frame_counter += 1
        if self.skip_frames > 1 and (self._frame_counter % self.skip_frames) != 0 and self._last_boxes:
            return self._last_boxes, self._last_probs
        try:
            # Detect on downscaled image
            boxes, probs = self.mtcnn.detect(small)
        except RuntimeError as e:
            if self._is_mps_pool_error(e) and self.device.type == 'mps':
                # Fallback to CPU due to MPS pooling issue
                print('MPS pooling error detected, switching FaceDetector to CPU')
                self.device = torch.device('cpu')
                self.mtcnn = MTCNN(keep_all=self.mtcnn.keep_all, device=self.device, thresholds=self.mtcnn.thresholds)
                # Retry detect on downscaled image
                boxes, probs = self.mtcnn.detect(small)
            else:
                raise
        if boxes is None:
            # clear cache if no faces found
            self._last_boxes = []
            self._last_probs = []
            return [], []
        # Scale boxes back to original frame size
        scale = 1 / self.resize_factor
        boxes_scaled = []
        for box in boxes:
            x1, y1, x2, y2 = [int(coord * scale) for coord in box]
            boxes_scaled.append((x1, y1, x2, y2))
        # cache for skipped frames
        self._last_boxes = boxes_scaled
        self._last_probs = probs.tolist()
        return boxes_scaled, probs.tolist()

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
