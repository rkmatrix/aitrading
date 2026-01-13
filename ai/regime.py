from collections import deque
import numpy as np

class RegimeDetector:
    def __init__(self, maxlen=60):
        self.window = deque(maxlen=maxlen)  # last 60 steps ~ 10 min if 10s loop

    def update(self, price):
        if price is not None and price > 0:
            self.window.append(float(price))

    def volatility(self):
        if len(self.window) < 5: return 0.0
        arr = np.array(self.window)
        return float(np.std(np.diff(np.log(arr))) * np.sqrt(252*6.5*60))  # rough annualized intraday

    def regime(self):
        vol = self.volatility()
        if vol > 0.6: return "HIGH_VOL"
        if vol > 0.3: return "MID_VOL"
        return "LOW_VOL"
