import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QDialog, QListWidget, QListWidgetItem, QFileDialog, QInputDialog, QMessageBox, QProgressDialog, QCheckBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from modules.video_capture import VideoCaptureManager
from modules.face_detection import FaceDetector
from modules.face_recognition import FaceRecognizer
from modules.database import DatabaseManager
from modules.blur import gaussian_blur_roi
import time
import numpy as np


class UserManagementDialog(QDialog):
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Manage Whitelist')
        self.db = DatabaseManager(db_path)
        self.detector = FaceDetector(resize_factor=0.5, skip_frames=1)
        self.recognizer = FaceRecognizer(db_path=None)

        # List widget
        self.listWidget = QListWidget()
        self.load_users()

        # Buttons
        self.addBtn = QPushButton('Add User')
        self.delBtn = QPushButton('Delete User')
        self.closeBtn = QPushButton('Close')
        self.delBtn.setEnabled(False)

        # Layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.addBtn)
        btn_layout.addWidget(self.delBtn)
        btn_layout.addWidget(self.closeBtn)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.listWidget)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        # Signals
        self.addBtn.clicked.connect(self.add_user)
        self.delBtn.clicked.connect(self.delete_user)
        self.closeBtn.clicked.connect(self.accept)
        self.listWidget.currentItemChanged.connect(lambda cur, prev: self.delBtn.setEnabled(cur is not None))

    def load_users(self):
        self.listWidget.clear()
        for u in self.db.list_users():
            item = QListWidgetItem(u['name'])
            item.setData(Qt.UserRole, u['id'])
            self.listWidget.addItem(item)

    def add_user(self):
        # Step 1: Prompt user name
        name, ok = QInputDialog.getText(self, 'User Name', 'Enter new user name:')
        if not ok or not name:
            return
        # Setup scanning preview for 5 seconds
        samples = []
        last_face = None
        duration = 7
        # Countdown before scanning
        cap = VideoCaptureManager()
        cv2.namedWindow('Scanning Face', cv2.WINDOW_NORMAL)
        for sec in (3, 2, 1):
            frame = cap.read()
            cv2.putText(frame, f"Starting in {sec}...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 3)
            cv2.imshow('Scanning Face', frame)
            cv2.waitKey(1000)
        # Begin scanning
        start = time.time()
        while time.time() - start < duration:
            frame = cap.read()
            boxes, _ = self.detector.detect(frame)
            if boxes:
                x1, y1, x2, y2 = boxes[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                face_img = frame[y1:y2, x1:x2]
                last_face = face_img
                try:
                    samples.append(self.recognizer.get_embedding(face_img))
                except Exception:
                    pass
            rem = max(0, duration - (time.time() - start))
            cv2.putText(frame, f"Scanning... {rem:.1f}s", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.imshow('Scanning Face', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyWindow('Scanning Face')
        if not samples:
            QMessageBox.warning(self, 'Error', 'No faces detected during scan')
            return
        # Show preview of the last captured face
        if last_face is not None:
            cv2.imshow('Preview Face', last_face)
            cv2.waitKey(3000)
            cv2.destroyWindow('Preview Face')
        # average embeddings and save
        emb = np.mean(np.stack(samples), axis=0)
        uid = self.db.add_user(name, emb)
        QMessageBox.information(self, 'Added', f"User '{name}' added (ID: {uid})")
        self.load_users()

    def delete_user(self):
        item = self.listWidget.currentItem()
        if not item:
            return
        uid = item.data(Qt.UserRole)
        name = item.text()
        if QMessageBox.question(self, 'Confirm', f"Delete '{name}'?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self.db.delete_user(uid)
        self.load_users()


class MainWindow(QWidget):
    def __init__(self, db_path='blurguard.db'):
        super().__init__()
        self.setWindowTitle('BlurGuard - Privacy-Aware AI Camera')
        # Widgets
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.start_btn = QPushButton('Start')
        self.stop_btn = QPushButton('Stop')
        self.manage_btn = QPushButton('Manage Whitelist')
        self.stop_btn.setEnabled(False)

        # Layouts
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.manage_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        # Pipelines
        self.cap_mgr = VideoCaptureManager()
        self.detector = FaceDetector(resize_factor=0.25, skip_frames=2)
        self.recognizer = FaceRecognizer(db_path=db_path)

        # Timer for live update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Connections
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.manage_btn.clicked.connect(self.manage_users)
        # Recording controls
        self.record_btn = QPushButton('Record')
        self.recording = False
        self.writer = None
        btn_layout.addWidget(self.record_btn)
        self.record_btn.clicked.connect(self.toggle_recording)
        # Overlay toggle
        self.overlay_check = QCheckBox('Overlay')
        self.overlay_check.setChecked(True)
        btn_layout.addWidget(self.overlay_check)

    def toggle_recording(self):
        """Start/stop video recording with blur applied"""
        if not self.recording:
            # Ask file path
            path, _ = QFileDialog.getSaveFileName(self, 'Save Video', '', 'MP4 Files (*.mp4)')
            if not path:
                return
            # Initialize writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            w = int(self.cap_mgr.width)
            h = int(self.cap_mgr.height)
            fps = 30.0
            self.writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            self.recording = True
            self.record_btn.setText('Stop Recording')
        else:
            # Stop recording
            self.recording = False
            if self.writer:
                self.writer.release()
                self.writer = None
            self.record_btn.setText('Record')

    def start(self):
        self.timer.start(16)  # ~60 FPS
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop(self):
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def manage_users(self):
        # Open whitelist management dialog
        dlg = UserManagementDialog(db_path='blurguard.db', parent=self)
        dlg.exec_()
        # Refresh recognizer with updated whitelist embeddings
        try:
            self.recognizer._load_known()
        except Exception:
            pass

    def update_frame(self):
        frame = self.cap_mgr.read()
        # Always detect faces and process blur/overlay
        boxes, probs = self.detector.detect(frame)
        valid = []
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            valid.append((x1, y1, x2, y2, crop, probs[idx]))
        if valid:
            boxes_f, face_imgs, confs = zip(*[((x1, y1, x2, y2), crop, conf) for x1, y1, x2, y2, crop, conf in valid])
            results = self.recognizer.recognize_faces(face_imgs)
            for (box, res, conf) in zip(boxes_f, results, confs):
                x1, y1, x2, y2 = box
                # Always blur unknown faces
                if res['id'] is None:
                    gaussian_blur_roi(frame, (x1, y1, x2, y2))
                # Overlay (box, name, confidence) when enabled
                if self.overlay_check.isChecked():
                    # Display detection confidence as integer percent
                    conf_text = f"{int(conf * 100)}%"
                    cv2.putText(frame, conf_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    if res['id'] is not None:
                        # Known face: draw box and name
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, res['name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Convert to QImage and display
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qt_img)
        # Write frame to file if recording
        if self.recording and self.writer:
            # cv2 expects BGR frame
            self.writer.write(frame)

        self.video_label.setPixmap(pix)

    def closeEvent(self, event):
        self.timer.stop()
        self.cap_mgr.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
