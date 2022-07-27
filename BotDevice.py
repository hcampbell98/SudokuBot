import cv2
import numpy as np
from ppadb.client import Client

class BotDevice():
    def __init__(self, adb, index) -> None:
        self.device = adb.devices()[index]
    
    def screenshot(self):
        rawScreenshot = self.device.screencap()
        return cv2.imdecode(np.frombuffer(rawScreenshot, dtype=np.uint8), 1)