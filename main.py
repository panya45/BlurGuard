#!/usr/bin/env python3
"""
Main application: real-time capture, detect, recognize, blur unknown faces.
"""
import cv2
import sys
from modules.video_capture import VideoCaptureManager
from modules.face_detection import FaceDetector
from modules.face_recognition import FaceRecognizer
from modules.blur import gaussian_blur_roi  # add import
from ui import main as ui_main  # add UI import


def main(db_path='blurguard.db'):
    # Initialize video capture, detector, recognizer
    cap_mgr = VideoCaptureManager()
    detector = FaceDetector(resize_factor=0.25, skip_frames=2)
    recognizer = FaceRecognizer(db_path=db_path)

    print("Starting live BlurGuard. Press 'q' to quit.")
    try:
        while True:
            frame = cap_mgr.read()
            # Face detection
            boxes, _ = detector.detect(frame)
            # Filter valid ROIs (non-zero size)
            valid = []
            for box in boxes:
                x1, y1, x2, y2 = box
                # skip invalid coordinates
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                valid.append((box, crop))
            if valid:
                boxes_f, face_imgs = zip(*valid)
                # Face recognition
                results = recognizer.recognize_faces(face_imgs)
                # Process each face
                for (box, res) in zip(boxes_f, results):
                    x1, y1, x2, y2 = box
                    if res['id'] is None:
                        # Unknown -> blur ROI using helper
                        gaussian_blur_roi(frame, (x1, y1, x2, y2))
                    else:
                        # Known -> draw box + name
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, res['name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Show frame
            cv2.imshow('BlurGuard', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        cap_mgr.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Use CLI if '--cli' flag provided, otherwise launch UI
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        main()
    else:
        ui_main()
