import argparse
import math
import random
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

INDEX_TIP = 8
THUMB_TIP = 4
MIDDLE_MCP = 9

STYLES = [
    {"name": "GLASS",    "kind": "glass",    "tint": (255, 210, 120), "border": (255, 255, 255), "spark": "dust"},
    {"name": "HOLOGRAM", "kind": "holo",     "tint": (255, 255, 140), "border": (220, 255, 255), "spark": "dust"},
    {"name": "INFERNO",  "kind": "glass",    "tint": (40, 120, 255),  "border": (90, 190, 255),  "spark": "embers"},
    {"name": "VOID",     "kind": "void",     "tint": (200, 80, 180),  "border": (255, 180, 255), "spark": "stars"},
    {"name": "PIXEL",    "kind": "pixel",    "tint": (150, 255, 120), "border": (190, 255, 210), "spark": "dust"},
    {"name": "X-RAY",    "kind": "negative", "tint": (235, 235, 235), "border": (255, 255, 255), "spark": "stars"},
]


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    print(f"Downloading model to {path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as exc:
        sys.exit(
            f"Could not download the model ({exc}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {path}"
        )
    print("Model ready.")
    return path


def build_detector(model_path: Path, max_hands: int = 2,
                   min_confidence: float = 0.5) -> "vision.HandLandmarker":
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def portal_corners(hands, width: int, height: int):
    if len(hands) < 2:
        return None
    hands = sorted(hands[:2], key=lambda h: h[0].x)    
    left, right = hands

    def px(lm):
        return np.array([lm.x * width, lm.y * height], dtype=np.float32)

    return np.array([
        px(left[INDEX_TIP]),    
        px(right[INDEX_TIP]),   
        px(right[THUMB_TIP]),    
        px(left[THUMB_TIP]),     
    ], dtype=np.float32)


class SmoothedQuad:
    def __init__(self, alpha: float = 0.45):
        self.alpha = alpha
        self.value = None

    def update(self, quad):
        if quad is None:
            self.value = None
            return None
        if self.value is None or self.alpha >= 1.0:
            self.value = quad.copy()
        else:
            self.value = self.value * (1.0 - self.alpha) + quad * self.alpha
        return self.value


class PinchTap:
    def __init__(self, pinch: float = 0.45, release: float = 0.75,
                 cooldown: float = 0.25):
        self.pinch = pinch
        self.release = max(release, pinch + 0.05)
        self.cooldown = cooldown
        self.pinched = False
        self.last = -1e9        

    @staticmethod
    def _ratio(hand) -> float:
        scale = math.hypot(hand[0].x - hand[MIDDLE_MCP].x,
                           hand[0].y - hand[MIDDLE_MCP].y) or 1e-6
        gap = math.hypot(hand[THUMB_TIP].x - hand[INDEX_TIP].x,
                         hand[THUMB_TIP].y - hand[INDEX_TIP].y)
        return gap / scale

    def update(self, hands, now: float) -> bool:
        if not hands:
            self.pinched = False
            return False
        gap = min(self._ratio(h) for h in hands)

        if self.pinched:
            if gap > self.release:
                self.pinched = False
            return False

        if gap < self.pinch and now - self.last > self.cooldown:
            self.pinched = True
            self.last = now
            return True
        return False


class Portal:
    def __init__(self, width: int, height: int, glass: float = 0.8,
                 sparkles: int = 70):
        self.w, self.h = width, height
        self.glass = min(max(glass, 0.0), 1.0)
        self.sparkles = []
        rng = random.Random(7)
        for _ in range(sparkles):
            self.sparkles.append((
                rng.random(),                      
                rng.random(),                       
                rng.uniform(0.4, 1.5),              
                rng.uniform(0.0, math.tau),         
                rng.uniform(0.2, 0.55),             
            ))
    def _quad_uv(self, quad, u, v):
        
        tl, tr, br, bl = quad
        top = tl * (1 - u) + tr * u
        bottom = bl * (1 - u) + br * u
        p = top * (1 - v) + bottom * v
        return int(p[0]), int(p[1])
    def _panel(self, frame, quad, style, now):
        h, w = frame.shape[:2]
        tint_layer = np.empty_like(frame)
        tint_layer[:] = style["tint"]
        kind = style["kind"]

        if kind == "glass":
            base = cv2.addWeighted(frame, 1.15, tint_layer, 0.35, -20)
            panel = cv2.addWeighted(base, 0.72, tint_layer, 0.28, 0)
            panel[::5] = cv2.addWeighted(panel[::5], 0.7, tint_layer[::5], 0.3, 6)

        elif kind == "holo":
            panel = cv2.addWeighted(frame, 0.85, tint_layer, 0.45, 15)
            panel[::3] = cv2.addWeighted(panel[::3], 0.55, tint_layer[::3], 0.45, 10)
            # moving bright scan band
            y = int((now * 140) % (h + 80)) - 40
            y0, y1 = max(y, 0), min(y + 26, h)
            if y1 > y0:
                panel[y0:y1] = cv2.addWeighted(panel[y0:y1], 1.0,
                                               tint_layer[y0:y1], 0.45, 0)

        elif kind == "void":
            panel = cv2.addWeighted(frame, 0.22, tint_layer, 0.30, 0)

        elif kind == "pixel":
            small = cv2.resize(frame, (max(w // 22, 8), max(h // 22, 8)),
                               interpolation=cv2.INTER_AREA)
            pixel = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            panel = cv2.addWeighted(pixel, 0.85, tint_layer, 0.22, 8)
            step = max(w // 22, 8)
            panel[::step] = cv2.addWeighted(panel[::step], 0.6,
                                            tint_layer[::step], 0.4, 0)
            panel[:, ::step] = cv2.addWeighted(panel[:, ::step], 0.6,
                                               tint_layer[:, ::step], 0.4, 0)

        else:  # negative / x-ray
            panel = cv2.addWeighted(255 - frame, 0.88, tint_layer, 0.12, 0)
            edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 140)
            panel[edges > 0] = (40, 40, 40)
        spark_layer = np.zeros_like(frame)
        glow_color = {
            "dust": (255, 255, 255),
            "embers": (60, 160, 255),
            "stars": (255, 240, 255),
        }[style["spark"]]

        for u, v, size, phase, speed in self.sparkles:
            if style["spark"] == "embers":
                v = (v - now * speed * 0.25) % 1.0        
            flick = 0.5 + 0.5 * math.sin(now * speed * 6.0 + phase)
            if flick < 0.35:
                continue
            x, y = self._quad_uv(quad, u, v)
            shade = tuple(int(c * flick) for c in glow_color)
            cv2.circle(spark_layer, (x, y), max(1, int(size)), shade, -1, cv2.LINE_AA)
        spark_layer = cv2.GaussianBlur(spark_layer, (0, 0), 2)
        cv2.add(panel, spark_layer, dst=panel)
        return panel
    def draw(self, frame, quad, style, now: float, show_glass: bool = True):
        pts = np.round(quad).astype(np.int32)

        if show_glass:
            mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
            cv2.fillConvexPoly(mask, pts, 255, cv2.LINE_AA)
            mask = cv2.GaussianBlur(mask, (0, 0), 4)

            panel = self._panel(frame, quad, style, now)
            m = (mask.astype(np.float32) / 255.0)[..., None] * self.glass
            blended = frame.astype(np.float32) * (1.0 - m) + panel.astype(np.float32) * m
            frame[:] = np.clip(blended, 0, 255).astype(np.uint8)

        border_layer = np.zeros_like(frame)
        cv2.polylines(border_layer, [pts], True, style["border"], 3, cv2.LINE_AA)
        for p in pts:                       
            cv2.circle(border_layer, tuple(p), 6, style["border"], -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(border_layer, (0, 0), 9)
        cv2.add(frame, glow, dst=frame)
        cv2.add(frame, border_layer, dst=frame)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, has_portal, style_name, flash_until, now, fps):
    h, w = frame.shape[:2]
    title = "P O R T A L   H A N D S"
    size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
    x = (w - size[0]) // 2
    cv2.putText(frame, title, (x, 38), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, title, (x, 38), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # current portal type
    cv2.putText(frame, f"PORTAL: {style_name}", (14, 74), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"PORTAL: {style_name}", (14, 74), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 240, 140), 2, cv2.LINE_AA)

    hint = ("frame with both hands   |   pinch = change portal"
            if has_portal else "show BOTH hands to open the portal")
    cv2.putText(frame, hint, (14, h - 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, hint, (14, h - 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (240, 240, 240), 1, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:4.1f} FPS   |   1-6 pick   g panel   s save   q quit",
                (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (200, 200, 200), 1, cv2.LINE_AA)
    if now < flash_until:
        text = style_name
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)[0]
        x = (w - size[0]) // 2
        cv2.putText(frame, text, (x, h // 2 + 10), cv2.FONT_HERSHEY_DUPLEX,
                    1.4, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(frame, text, (x, h // 2 + 10), cv2.FONT_HERSHEY_DUPLEX,
                    1.4, (255, 255, 255), 2, cv2.LINE_AA)


def save_shot(image) -> Path:
    name = f"portal_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
    cv2.imwrite(name, image)
    return Path(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-framed glowing portal")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="detection/tracking confidence 0-1 (default 0.5)")
    parser.add_argument("--glass", type=float, default=0.8,
                        help="panel strength 0..1 (default 0.8)")
    parser.add_argument("--pinch", type=float, default=0.45,
                        help="pinch distance that counts as a tap (default 0.45, "
                             "lower = fingers must touch)")
    parser.add_argument("--still", action="store_true",
                        help="no smoothing: the portal snaps to your hands")
    args = parser.parse_args()

    detector = build_detector(ensure_model(), 2, args.confidence)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(
            f"Could not open camera {args.camera}. "
            "Check that it is connected, not used by another app, "
            "and that the terminal has camera permission."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    window = "Portal Hands (q to quit)"
    start = time.monotonic()
    prev = start
    fps = 0.0

    quad_filter = SmoothedQuad(alpha=1.0 if args.still else 0.45)
    pinch_tap = PinchTap(pinch=args.pinch)
    portal = None
    style_index = 0
    flash_until = 0.0
    show_panel = True

    with detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break

            frame = cv2.flip(frame, 1)        
            h, w = frame.shape[:2]
            if portal is None:
                portal = Portal(w, h, args.glass)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now = time.monotonic()
            result = detector.detect_for_video(mp_image, int((now - start) * 1000))
            hands = list(result.hand_landmarks)

            quad = quad_filter.update(portal_corners(hands, w, h))
            if quad is not None and cv2.contourArea(quad.astype(np.float32)) < 1500:
                quad = None          

            # pinch thumb + index together -> next portal type
            tapped = pinch_tap.update(hands, now) if len(hands) >= 2 else False
            if tapped:
                style_index = (style_index + 1) % len(STYLES)
                flash_until = now + 1.2
                print(f"Portal: {STYLES[style_index]['name']}")

            style = STYLES[style_index]
            if quad is not None:
                portal.draw(frame, quad, style, now, show_panel)

            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            draw_hud(frame, quad is not None, style["name"], flash_until, now, fps)

            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("6"):
                style_index = min(key - ord("1"), len(STYLES) - 1)
                flash_until = now + 1.2
            if key == ord("g"):
                show_panel = not show_panel
            if key == ord("s"):
                print(f"Saved {save_shot(frame)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()