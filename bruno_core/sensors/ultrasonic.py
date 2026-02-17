import sys, time
from typing import List, Optional
import numpy as np
sys.path.append('/home/pi/MasterPi')
try:
    import pandas as pd; PANDAS_OK=True
except Exception:
    pd=None; PANDAS_OK=False
import common.sonar as Sonar
class UltrasonicRGB:
    def __init__(self, avg_samples:int=5, sample_delay:float=0.02):
        self.sonar=Sonar.Sonar(); self.avg_samples=avg_samples; self.sample_delay=sample_delay; self.set_rgb(0,0,255)
    def get_distance_cm(self)->Optional[float]:
        vals:List[float]=[]
        for _ in range(self.avg_samples):
            try:
                d_cm=self.sonar.getDistance()/10.0
                if 2.0<=d_cm<=400.0: vals.append(float(d_cm))
            except Exception: pass
            time.sleep(self.sample_delay)
        if not vals: return None
        if len(vals)>=4: vals.remove(max(vals)); vals.remove(min(vals))
        if PANDAS_OK and len(vals)>=3:
            s=pd.Series(vals); m,sd=float(s.mean()),float(s.std())
            if sd>0: s=s[np.abs(s-m)<=sd]
            if len(s)>0: return float(s.mean())
        return float(np.mean(vals))
    def set_rgb(self,r:int,g:int,b:int):
        try: self.sonar.setPixelColor(0,(r,g,b)); self.sonar.setPixelColor(1,(r,g,b))
        except Exception: pass
