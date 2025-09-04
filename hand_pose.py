import cv2
import numpy as np
import tkinter as tk
from cvzone.HandTrackingModule import HandDetector

# ---------------------------
# Hand Detector
# ---------------------------
detector = HandDetector(detectionCon=0.8, maxHands=2)

# ---------------------------
# Button Class
# ---------------------------
class Button:
    def __init__(self, pos, text):
        self.pos = pos
        self.size = (140, 40)
        self.text = text
        self.active = False

    def draw(self, img):
        x, y = self.pos
        color = (0, 200, 0) if self.active else (200, 200, 200)
        cv2.rectangle(img, (x, y), (x+self.size[0], y+self.size[1]), color, -1)
        cv2.rectangle(img, (x, y), (x+self.size[0], y+self.size[1]), (0,0,0), 2)
        cv2.putText(img, self.text, (x+10, y+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    def checkClick(self, x, y):
        bx, by = self.pos
        if bx < x < bx+self.size[0] and by < y < by+self.size[1]:
            return True
        return False

# ---------------------------
# Filters
# ---------------------------
def apply_invert(img):
    return cv2.bitwise_not(img)

def apply_rgb_shift(img):
    b, g, r = cv2.split(img)
    return cv2.merge([g, r, b])

def apply_old_money(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sepia = cv2.applyColorMap(gray, cv2.COLORMAP_PINK)
    return sepia

def apply_pixelate(img, scale=0.1):
    h, w = img.shape[:2]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    temp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_mario(img):
    div = 64  # quantization level
    return (img // div) * div

def apply_wavy(img):
    h, w = img.shape[:2]
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    map_x = map_x + 10 * np.sin(map_y / 20)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

# ---------------------------
# Setup Filters + Buttons
# ---------------------------
filters = {
    "Normal": lambda x: x,
    "Invert": apply_invert,
    "RGB": apply_rgb_shift,
    "OldMoney": apply_old_money,
    "Pixel": apply_pixelate,
    "Mario": apply_mario,
    "Wavy": apply_wavy
}

buttons = [Button((20 + i*150, 20), f) for i, f in enumerate(filters.keys())]
current_filter = "Normal"

# ---------------------------
# Get Screen Resolution
# ---------------------------
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# ---------------------------
# Main Loop
# ---------------------------
cap = cv2.VideoCapture(0)

cv2.namedWindow("Hand Filter", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Filter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    # Resize webcam feed to full screen
    img = cv2.resize(img, (screen_width, screen_height))

    # Detect hands
    hands, img = detector.findHands(img, draw=False)

    # Draw Buttons
    for button in buttons:
        button.draw(img)

    # Hand area polygon (between index + thumb of both hands)
    if hands and len(hands) == 2:
        pts = []
        for hand in hands:
            lm = hand["lmList"]
            pts.append(lm[4][:2])   # Thumb tip
            pts.append(lm[8][:2])   # Index tip
        pts = np.array(pts, np.int32)

        # Bounding box
        x, y, w, h = cv2.boundingRect(pts)
        roi = img[y:y+h, x:x+w]

        if roi.size > 0:
            filtered = filters[current_filter](roi)
            # Mask polygon
            mask = np.zeros_like(img[y:y+h, x:x+w])
            cv2.fillPoly(mask, [pts - [x, y]], (255, 255, 255))
            filtered_masked = cv2.bitwise_and(filtered, mask)
            bg = cv2.bitwise_and(img[y:y+h, x:x+w], 255 - mask)
            img[y:y+h, x:x+w] = cv2.add(bg, filtered_masked)

        # Polygon outline
        cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)

    # Check clicks
    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        if fingers == [0,1,0,0,0]:  # only index finger up
            cx, cy = hand["lmList"][8][:2]
            for button in buttons:
                if button.checkClick(cx, cy):
                    current_filter = button.text
                    for b in buttons:
                        b.active = (b == button)

    cv2.putText(img, f"Filter: {current_filter}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    cv2.imshow("Hand Filter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
