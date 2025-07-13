import cv2


def gaussian_blur_roi(frame, box, ksize=(101, 101)):
    """
    Apply Gaussian blur to the region of interest in the frame.
    frame: BGR numpy array
    box: tuple (x1, y1, x2, y2)
    ksize: blur kernel size (stronger blur)
    """
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, ksize, 0)
    frame[y1:y2, x1:x2] = blurred
    return frame


def pixelate_roi(frame, box, pixel_size=10):
    """
    Apply pixelation to the region of interest in the frame.
    frame: BGR numpy array
    box: tuple (x1, y1, x2, y2)
    pixel_size: size of pixel block
    """
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    # downscale and upscale for pixelation
    temp = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = pixelated
    return frame
