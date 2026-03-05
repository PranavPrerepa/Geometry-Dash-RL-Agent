import time
import cv2
import pyautogui
import numpy as np

# Disable failsafe since the game window might be near the corner of the screen
pyautogui.FAILSAFE = False

# Import our screen capture logic
from screen_capture import get_window_rect, mss

class GeometryDashEnv:
    def __init__(self):
        """
        An OpenAI Gym-like environment for Geometry Dash.
        """
        # Get window position so we know where to click
        self.monitor = get_window_rect("Geometry Dash")
        if not self.monitor:
            print("WARNING: Geometry Dash window not found. Using default Region.")
            self.monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
            
        # We need the center of the window to send click events
        self.center_x = self.monitor["left"] + (self.monitor["width"] // 2)
        self.center_y = self.monitor["top"] + (self.monitor["height"] // 2)
        
        self.sct = mss.mss()
        
        # Load the "Attempt" template for death detection
        try:
            # We assume you'll save a small crop of the "Attempt" text as 'attempt_template.png'
            self.death_template = cv2.imread("attempt_template.png", cv2.IMREAD_GRAYSCALE)
            if self.death_template is not None:
                self.template_w, self.template_h = self.death_template.shape[::-1]
            else:
                print("WARNING: 'attempt_template.png' not found. Death detection will always be False until created.")
        except Exception as e:
            print(f"Template load error: {e}")
            self.death_template = None

        # Actions: 0 = Do Nothing, 1 = Jump (Hold)
        self.action_space_size = 2
        
        # To avoid holding forever, we track if we are currently holding
        self.is_holding = False
        
        # Store a short history of frames to detect if the screen becomes static (death screen)
        self.frame_history = []
        self.steps_taken = 0

    def _get_frame(self):
        """Captures the screen and returns a 1x128x128 normalized grayscale numpy array."""
        sct_img = self.sct.grab(self.monitor)
        img = np.array(sct_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        
        # Keep a history of the last 5 frames instead of 15 to detect death faster
        self.frame_history.append(gray)
        if len(self.frame_history) > 5:
            self.frame_history.pop(0)
            
        # Check for death using OpenCV template OR static frames
        done = self._check_death(gray)
        
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        
        # Normalize to 0-1 and add channel dimension (C, H, W) -> (1, 128, 128)
        normalized = resized.astype(np.float32) / 255.0
        frame = np.expand_dims(normalized, axis=0)
        
        return frame, done

    def _check_death(self, frame):
        """Uses OpenCV to compare frames and detect if the screen has stopped scrolling (death)."""
        # If the screen is completely static for 5 frames, we hit the death screen.
        # But we only check this after taking some steps, so we don't instantly die on spawn.
        if self.steps_taken > 5 and len(self.frame_history) == 5:
            first_frame_in_hist = self.frame_history[0].astype(np.float32)
            current_frame = frame.astype(np.float32)
            
            # MSE between the frame from ~0.75s ago and now
            err = np.sum((first_frame_in_hist - current_frame) ** 2) / float(frame.size)
            
            # If the error is extremely low, the screen stopped scrolling
            if err < 5.0:
                print(f"Death detected! Screen is static (MSE: {err:.2f})")
                return True
                
        # Fallback Template Matching Detection 
        if self.death_template is not None:
            res = cv2.matchTemplate(frame, self.death_template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8 
            loc = np.where(res >= threshold)
            if len(loc[0]) > 0:
                print("Death detected via template matching!")
                return True
                
        return False

    def reset(self):
        """
        Resets the environment (e.g., after dying). 
        Will explicitly click the level retry button to get past the death screen.
        """
        # Release mouse
        if self.is_holding:
            pyautogui.mouseUp(button='left')
            self.is_holding = False
            
        print("Resetting environment... clicking Retry.")
        
        # Click the 'Retry' button on the death screen (approx 28% from left, 83% from top)
        retry_x = int(self.monitor["left"] + self.monitor["width"] * 0.28)
        retry_y = int(self.monitor["top"] + self.monitor["height"] * 0.83)
        pyautogui.click(x=retry_x, y=retry_y)
        
        # Also press spacebar as a secondary fallback to start the level
        time.sleep(0.1)
        pyautogui.press('space')
        
        # Give the game time to finish the restart transition and begin scrolling
        time.sleep(1.0) 
        
        # Clear frame history and steps counter for the new episode
        self.frame_history = []
        self.steps_taken = 0
        
        frame, _ = self._get_frame()
        return frame

    def step(self, action):
        """
        Executes an action, waits a tiny bit, and returns the new state and reward.
        
        Args:
            action (int): 0 for Nothing, 1 for Jump/Hold
        
        Returns:
            next_state (numpy.ndarray): The new 1x128x128 frame
            reward (float): The reward for this step
            done (bool): Whether the agent died
        """
        # Execute Action
        if action == 1:
            if not self.is_holding:
                # Need to click exactly inside the Geometry Dash window
                pyautogui.mouseDown(x=self.center_x, y=self.center_y, button='left')
                self.is_holding = True
        elif action == 0:
            if self.is_holding:
                pyautogui.mouseUp(button='left')
                self.is_holding = False

        # Wait a fraction of a second to let the action affect the game
        # For 60FPS it would be ~0.016s, but standard DQN often uses frame skipping (e.g. 4 frames = ~0.06s)
        time.sleep(0.05) 
        
        self.steps_taken += 1
        
        # Observe result
        next_state, done = self._get_frame()
        
        # Calculate Reward
        if done:
            reward = -100.0  # Massive penalty for dying
        else:
            reward = 1.0     # Small continuous reward for surviving
            
        return next_state, reward, done

# Quick test if run directly
if __name__ == "__main__":
    env = GeometryDashEnv()
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Take a random test step
    next_state, reward, done = env.step(action=1)
    print(f"Step Result - Reward: {reward}, Died: {done}")
