"""
Simple test for VideoCaptureManager
"""
from modules.video_capture import VideoCaptureManager

def main():
    print("Initializing VideoCaptureManager...")
    cap_mgr = VideoCaptureManager()
    try:
        frame = cap_mgr.read()
        print(f"Captured frame of shape: {frame.shape}")
    except Exception as e:
        print(f"Error during capture: {e}")
    finally:
        cap_mgr.release()
        print("Released camera.")

if __name__ == '__main__':
    main()
