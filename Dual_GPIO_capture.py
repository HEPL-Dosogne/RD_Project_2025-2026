import Jetson.GPIO as gpio
import cv2
import time
import os
from datetime import datetime
from threading import Thread, Event

trigger_pin = 40
cam_indices = [0, 1, 2]

# Dossier d'enregistrement
save_dir = "captures"
os.makedirs(save_dir, exist_ok=True)

def open_cam(device_id):
    pipeline = f"v4l2src device=/dev/video{device_id} ! videoconvert ! appsink"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"Caméra {device_id} indisponible")
        return None
    print(f"Caméra {device_id} initialisée")
    return cap


class CameraThread(Thread):
    def __init__(self, cam_id):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.cap = open_cam(cam_id)
        self.capture_event = Event()
        self.running = True

    def run(self):
        while self.running:
            if self.capture_event.is_set():
                ret, frame = self.cap.read()
                if ret:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(save_dir, f"cam{self.cam_id}_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Image enregistrée : {filename}")
                self.capture_event.clear()
            time.sleep(0.04)  # 20Hz = 1 image/50ms

    def trigger(self):
        self.capture_event.set()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

# Ouvrir les caméras
cams = [CameraThread(i) for i in cam_indices]
for cam in cams:
    cam.start()

gpio.setmode(gpio.BOARD)
gpio.setup(trigger_pin, gpio.IN)

print("Système prêt sur GPIO40")

try:
    while True:
        gpio.wait_for_edge(trigger_pin, gpio.RISING)
        for cam in cams:
            cam.trigger()

except KeyboardInterrupt:
    print("\nArrêt du test")
finally:
    for cam in cams:
        cam.stop()
    gpio.cleanup()

