import time
class Snapshotter:
    def __init__(self, interval_s:int):
        self.interval_s=interval_s; self.last_t=0.0; self.count=0
    def due(self)->bool:
        now=time.time();
        return True if self.count==0 else (now-self.last_t)>=self.interval_s
    def mark(self):
        self.last_t=time.time(); self.count+=1
