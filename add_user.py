#!/usr/bin/env python3
"""
Script to capture a face, generate embedding, and add a user to the whitelist DB.
"""
from modules.video_capture import VideoCaptureManager
from modules.face_detection import FaceDetector
from modules.face_recognition import FaceRecognizer
from modules.database import DatabaseManager
import cv2
import time
import numpy as np

def add_user(name, db_path='blurguard.db'):
    # Initialize components
    cap_mgr = VideoCaptureManager()
    detector = FaceDetector()
    recognizer = FaceRecognizer(db_path=db_path)
    db = recognizer.db if recognizer.db else DatabaseManager(db_path)

    # Capture face samples for 5 seconds with live preview
    duration = 5
    samples = []
    print(f"Scanning face for {duration} seconds. Please look at the camera...")
    cv2.namedWindow('Scanning Face', cv2.WINDOW_NORMAL)
    t0 = time.time()
    while True:
        frame = cap_mgr.read()
        elapsed = time.time() - t0
        if elapsed > duration:
            break
        # detect face and draw box
        boxes, _ = detector.detect(frame)
        if boxes:
            x1, y1, x2, y2 = boxes[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            face_img = frame[y1:y2, x1:x2]
            try:
                emb = recognizer.get_embedding(face_img)
                samples.append(emb)
            except Exception:
                pass
        # overlay countdown
        remaining = max(0, duration - elapsed)
        cv2.putText(frame, f"Scanning... {remaining:.1f}s", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow('Scanning Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap_mgr.release()
    cv2.destroyWindow('Scanning Face')
    if not samples:
        print("No faces detected during capture. Please try again.")
        return
    # Average embeddings for stability
    emb = np.mean(np.stack(samples), axis=0)

    # Optionally show last captured face for confirmation
    cv2.imshow('Sampled Face', face_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # Add to database
    user_id = db.add_user(name, emb)
    print(f"User '{name}' added with ID: {user_id}")

if __name__ == '__main__':
    name = input('Enter new user name: ')
    add_user(name)
