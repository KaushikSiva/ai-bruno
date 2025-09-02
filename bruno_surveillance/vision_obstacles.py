from typing import Dict, List
import numpy as np
import cv2

def _estimate_px_distance(w: int, h: int, y: int, H: int) -> float:
    size = (w * h) / (100 * 100)
    pos  = (H - y) / max(H, 1)
    prox = size * 0.6 + pos * 0.4
    return max(20, 200 - prox * 100)

def vision_obstacles(frame, cfg: Dict) -> List[Dict]:
    try:
        H, W = frame.shape[:2]
        r = cfg.get('roi', {'top':0.5, 'bottom':0.95, 'left':0.15, 'right':0.85})
        y1, y2 = int(H*r['top']), int(H*r['bottom'])
        x1, x2 = int(W*r['left']), int(W*r['right'])
        roi = frame[y1:y2, x1:x2]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        edges = cv2.Canny(g, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg.get('edge_min_area', 800):
                continue
            x, y, w, h = cv2.boundingRect(c)
            fy = y + y1
            dpx = _estimate_px_distance(w, h, fy, H)
            cx = x + x1 + w / 2.0
            ang = np.degrees(np.arctan2(cx - W/2.0, W/2.0))
            if abs(ang) < 60:
                obs.append({'bbox': (x + x1, y + y1, w, h), 'px': dpx, 'angle': ang})
        return obs
    except Exception:
        return []
