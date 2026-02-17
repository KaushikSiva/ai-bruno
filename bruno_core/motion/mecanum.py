import sys, time
sys.path.append('/home/pi/MasterPi')
import common.mecanum as mecanum
class MecanumWrapper:
    def __init__(self, forward_speed:int=40, turn_speed:int=40):
        self.car=mecanum.MecanumChassis(); self.forward_speed=forward_speed; self.turn_speed=turn_speed
    def set_velocity(self, speed: float, direction_deg: float, rotation: float):
        try: self.car.set_velocity(speed, direction_deg, rotation)
        except Exception: pass
    def stop(self):
        try: self.car.set_velocity(0,0,0)
        except Exception: pass
    def forward(self, duration:float=0.0):
        try:
            self.car.set_velocity(self.forward_speed,90,0)
            if duration>0: time.sleep(duration); self.stop()
        except Exception: pass
    def reverse_burst(self, duration:float=0.6):
        try: self.car.set_velocity(-self.forward_speed,90,0); time.sleep(duration)
        except Exception: pass
        finally: self.stop()
    def turn_left(self, duration:float):
        try: self.car.set_velocity(0,90,-0.5); time.sleep(duration)
        except Exception: pass
        finally: self.stop()
    def turn_right(self, duration:float):
        try: self.car.set_velocity(0,90,0.5); time.sleep(duration)
        except Exception: pass
        finally: self.stop()
