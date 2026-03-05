import time
import mss
import cv2
import numpy as np
import os
import subprocess

# Suppress Qt font warnings from OpenCV
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

def get_window_rect(window_name="Geometry Dash"):
    """Uses xdotool to find the window's position and size on Linux."""
    try:
        # Get the window ID
        result = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", window_name],
            capture_output=True, text=True, check=True
        )
        window_ids = result.stdout.strip().split('\n')
        if not window_ids or not window_ids[0]:
            print(f"Error: Window '{window_name}' not found.")
            return None
            
        win_id = window_ids[0]
        
        # Get the window geometry
        geom_result = subprocess.run(
            ["xdotool", "getwindowgeometry", win_id],
            capture_output=True, text=True, check=True
        )
        
        lines = geom_result.stdout.split('\n')
        pos_line = next((line for line in lines if "Position:" in line), None)
        geom_line = next((line for line in lines if "Geometry:" in line), None)
        
        if pos_line and geom_line:
            pos_str = pos_line.split(":")[1].split("(")[0].strip()
            left, top = map(int, pos_str.split(","))
            
            geom_str = geom_line.split(":")[1].strip()
            width, height = map(int, geom_str.split("x"))
            
            return {"top": top, "left": left, "width": width, "height": height}
    except FileNotFoundError:
        print("Error: 'xdotool' is not installed. Please run: sudo apt install xdotool")
    except Exception as e:
        print(f"Failed to get window geometry: {e}")
    
    return None

def capture_screen():
    print("Looking for 'Geometry Dash' window...")
    monitor = get_window_rect("Geometry Dash")
    
    # Fallback if window not found
    if not monitor:
        print("Falling back to absolute screen region...")
        monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

    # Desired FPS
    target_fps = 60
    target_frame_time = 1.0 / target_fps

    with mss.mss() as sct:
        print(f"Starting screen capture of region {monitor} at {target_fps} FPS...")
        print("Press 'q' in the display window or Ctrl+C in terminal to stop.")
        
        try:
            while True:
                start_time = time.perf_counter()

                # Grab the screen pixels
                sct_img = sct.grab(monitor)
                
                # Convert the raw pixels to a numpy array (format is BGRA)
                img = np.array(sct_img)
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                
                # Resize to 128x128
                resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

                # --- DO SOMETHING WITH THE IMAGE HERE ---
                # E.g., pass 'resized' to your RL model.
                
                # For demonstration, we display the result
                cv2.imshow("Geometry Dash Capture (128x128 Grayscale)", resized)
                
                # Press 'q' to quit (waitKey is also needed for the OpenCV window to update)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                # Calculate how much time the processing took and sleep the remainder to maintain ~60 FPS
                elapsed_time = time.perf_counter() - start_time
                sleep_time = target_frame_time - elapsed_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nCapture stopped by user.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_screen()
