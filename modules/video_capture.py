import cv2

class VideoCaptureManager:
    """
    Manages video capture, auto-selecting the Continuity Camera (iPhone) if no index specified.
    Uses AVFoundation backend on macOS.
    """
    def __init__(self, index=None, width=1920, height=1080, fps=60):
        self.width = width
        self.height = height
        self.fps = fps
        # Select camera index if not provided
        self.index = index if index is not None else self.select_continuity_camera()
        # Open capture with AVFoundation backend
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        # Set capture properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    @staticmethod
    def select_continuity_camera(max_index=5, width=1920, height=1080, fps=60):
        """
        Try to find a camera that supports the desired resolution and FPS.
        Returns the first index matching 1080p@30fps.
        """
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ret, frame = cap.read()
            if ret and frame is not None and frame.shape[1] == width and frame.shape[0] == height:
                cap.release()
                return idx
            cap.release()
        raise RuntimeError("Continuity camera not found")

    def read(self):
        """
        Read a frame from the camera. Returns a BGR numpy array.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame from camera index {self.index}")
        return frame

    def release(self):
        """
        Release the video capture device.
        """
        if self.cap:
            self.cap.release()
