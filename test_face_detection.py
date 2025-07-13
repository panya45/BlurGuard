"""
Test for FaceDetector using a captured frame.
"""
from modules.video_capture import VideoCaptureManager
from modules.face_detection import FaceDetector
import cv2

def main():
    cap_mgr = VideoCaptureManager()
    detector = FaceDetector()
    try:
        while True:
            frame = cap_mgr.read()
            boxes, probs = detector.detect(frame)
            # Draw detection results
            for box, prob in zip(boxes, probs):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{prob:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow('Face Detection Live', frame)
            # exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error in face detection loop: {e}")
    finally:
        cap_mgr.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
