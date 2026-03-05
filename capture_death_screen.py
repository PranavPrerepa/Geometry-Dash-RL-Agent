import time
import cv2
import numpy as np
from screen_capture import get_window_rect, mss
import os

def capture_attempt():
    monitor = get_window_rect("Geometry Dash")
    if not monitor:
        print("Geometry Dash not found. Please open the game first.")
        return

    print("--- TEMPLATE CAPTURE SCRIPT ---")
    print("This script will help you automatically create the 'attempt_template.png'")
    print("that the AI uses to know when it dies.\n")
    print("INSTRUCTIONS:")
    print("1. Tab into Geometry Dash and start playing.")
    print("2. Crash your ship/cube intentionally.")
    print("3. Ensure the 'Attempt X' text is fully visible on your screen when the timer hits 0.\n")
    
    print("Taking picture in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
        
    print("Capturing...")
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        
        # We crop a small rectangular region where the "Attempt" text pops up.
        # This usually occurs around the vertical middle and horizontal center.
        h, w = gray.shape
        crop_y1 = int(h * 0.35)
        crop_y2 = int(h * 0.50)
        crop_x1 = int(w * 0.35)
        crop_x2 = int(w * 0.65)
        
        template = gray[crop_y1:crop_y2, crop_x1:crop_x2]
        
        cv2.imwrite("attempt_template.png", template)
        print("\nSuccess! Saved 'attempt_template.png'.")
        print("You can now safely restart the training script.")

if __name__ == '__main__':
    capture_attempt()
