from io import BytesIO
import base64
import os
import sys

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

if __name__ == "__main__":
    try:
        from .update import ensure_ca_bundle, maybe_update
    except ImportError:
        from update import ensure_ca_bundle, maybe_update
    ensure_ca_bundle()
    maybe_update()
else:
    try:
        from .update import ensure_ca_bundle
    except ImportError:
        try:
            from update import ensure_ca_bundle
        except ImportError:
            ensure_ca_bundle = None
    if ensure_ca_bundle is not None:
        ensure_ca_bundle()

try:
    from .paddle_bootstrap import ensure_paddle
except ImportError:
    from paddle_bootstrap import ensure_paddle
ensure_paddle()

import obsws_python as obsws
from datetime import datetime
import requests
import traceback
import pygetwindow as gw
import mss
import asyncio
import websockets
import json
import gc
from paddleocr import PaddleOCR
import threading
import numpy as np
import cv2
try:
    from . import broadcast
    from . import dialog
except ImportError:
    import broadcast
    import dialog
import routines
from typing import Callable, Dict, List, Optional
from PIL import Image, ImageFile
import configparser
import time


if __name__ == "__main__":
    print("Initializing...")
    from routines import client_name
    try:
        from build_info import __version__  # type: ignore
    except Exception:
        __version__ = "DEV"
    print(f"Welcome to {client_name.upper()} - build: {__version__}")
    from routines import payload
ImageFile.LOAD_TRUNCATED_IMAGES = True

config = configparser.ConfigParser()
config.read('config.ini')
processing_message = False
reader = PaddleOCR(
    ocr_version="PP-OCRv6",
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
ocr_stats = {"calls": 0, "ms_total": 0.0, "ms_samples": []}
_OCR_SAMPLE_CAP = 20000
refresh_rate = config.getfloat('settings', 'refresh_rate')
capture_mode = config.get('settings', 'capture_mode')
executable_title = config.get('settings', 'executable_title', fallback="")
obs = None
base_height = 1080
base_width = 1920


def reset_ocr_stats() -> None:
    ocr_stats["calls"] = 0
    ocr_stats["ms_total"] = 0.0
    ocr_stats["ms_samples"] = []


def _note_ocr(ms: float) -> None:
    ocr_stats["calls"] += 1
    ocr_stats["ms_total"] += ms
    samples = ocr_stats["ms_samples"]
    if len(samples) < _OCR_SAMPLE_CAP:
        samples.append(ms)


def _paddle_result_dict(res):
    """PaddleOCR 3.x OCRResult is a dict with rec_texts. .json wraps that in {'res': ...}."""
    if isinstance(res, dict) and "rec_texts" in res:
        return res
    data = getattr(res, "json", res)
    if callable(data):
        data = data()
    if isinstance(data, dict) and isinstance(data.get("res"), dict):
        data = data["res"]
    return data if isinstance(data, dict) else {}


def _paddle_texts(raw, allowlist: str = None, low_text: float = 0.4):
    texts = []
    if not raw:
        return None
    for res in raw:
        data = _paddle_result_dict(res)
        rec_texts = data.get("rec_texts") or []
        rec_scores = list(data.get("rec_scores") or [])
        for i, text in enumerate(rec_texts):
            if not text:
                continue
            score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
            if score < low_text:
                continue
            if allowlist:
                text = "".join(c for c in text if c in allowlist)
            if text:
                texts.append(text)
    return texts or None


def print_with_time(*args, debug_only=False, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if debug_only and not config.getboolean('settings', 'debug_mode', fallback=False):
        return
    print(timestamp, "-", *args, **kwargs)


def capture_screen(payload):
    global obs
    if capture_mode == 'obs':
        try:
            if not obs:
                obs = obsws.ReqClient(
                    host=config.get('obs', 'host', fallback='localhost'),
                    port=config.get('obs', 'port', fallback=4455),
                    password=config.get('obs', 'password', fallback='')
                )
        except Exception:
            print_with_time("Could not connect to OBS. Retrying...")
            payload['state'] = None
            return None, 1.0, 1.0
        while True:
            try:
                response = obs.get_source_screenshot(
                    name=config.get('obs', 'source_title', fallback=""),
                    img_format="webp",
                    width=config.getint('obs', 'width', fallback=1920),
                    height=config.getint('obs', 'height', fallback=1080),
                    quality=95
                )
                prefix = "base64,"
                idx = response.image_data.find(prefix)
                img_str = (
                    response.image_data[idx + len(prefix):]
                    if idx != -1
                    else response.image_data
                )
                img_data = base64.b64decode(img_str)
                img = Image.open(BytesIO(img_data))
            except Exception as e:
                print_with_time(f"Error capturing screen from OBS: {e}")
                continue
            break
    elif capture_mode == 'game':
        capture_attempts = 0
        while True:
            windows = gw.getWindowsWithTitle(executable_title)
            if windows:
                window = windows[0]
                if capture_attempts > 0:
                    print(f"Found executable {executable_title}")
                break
            else:
                if capture_attempts < 1:
                    print(
                        f"Executable {executable_title} not found. Ensure it is running and visible.")
                capture_attempts += 1
                payload['state'] = None
                continue

        # Get the window's bounding box
        # Get the window's dimensions
        width = window.right - window.left
        height = window.bottom - window.top

        # Calculate target height for 16:9 aspect ratio
        target_height = int(width * (9 / 16))

        # If current height is larger than target, adjust top to crop from bottom
        if height > target_height:
            adjusted_top = window.bottom - target_height
        else:
            adjusted_top = window.top

        bbox = (window.left, adjusted_top, window.right, window.bottom)

        with mss.mss() as sct:
            # Capture the screen using the bounding box
            screenshot = sct.grab(bbox)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

    # also return the scale of the image based off base resolution (1080p)
    image_width, image_height = img.size
    scale_x = image_width / base_width
    scale_y = image_height / base_height
    return img, scale_x, scale_y


def is_within_deviation(pixel, target_color, deviation):
    return np.all(np.abs(np.array(pixel[:3] if type(pixel) is tuple else [pixel, pixel, pixel]) - np.array(target_color)) <= 255 * deviation)


def resize_template(template, scale_x, scale_y):
    h, w = template.shape[:2]
    return cv2.resize(template, (int(w * scale_x), int(h * scale_y)), interpolation=cv2.INTER_AREA)


def detect_image(img, scale_x, scale_y, template_file: str, region: tuple[int, int, int, int] = None):
    # Crop the specific area
    if region:
        x, y, w, h = region
        img = img.crop((x, y, x + w, y + h))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Load the template images
    template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)

    if template is None:
        raise FileNotFoundError("Template image not found")

    template = resize_template(template, scale_x, scale_y)

    # Perform template matching
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    del img
    gc.collect()
    return np.max(res)


def get_color_match_in_region(img, region: tuple[int, int, int, int], target_color: tuple | list[tuple], deviation: float = 0.15):
    x, y, w, h = region
    cropped_area = img.crop((x, y, x + w, y + h))
    width, height = cropped_area.size
    total_pixels = width * height

    # Ensure target_color is a list of tuples
    if isinstance(target_color, tuple) and not isinstance(target_color[0], (tuple, list)):
        target_color = [target_color]

    matches = {i: 0 for i in range(len(target_color))}

    for i in range(width):
        for j in range(height):
            pixel = cropped_area.getpixel((i, j))
            if isinstance(pixel, int):
                pixel = (pixel, pixel, pixel)
            for idx, color in enumerate(target_color):
                if is_within_deviation(pixel, color, deviation):
                    matches[idx] += 1
                    break  # Only count a pixel for the first matching color

    # Return match ratios for each color
    if len(matches) > 1:
        return {idx: count / total_pixels for idx, count in matches.items()}
    else:
        return list(matches.values())[0] / total_pixels


def remove_neighbor_duplicates(input_list):
    if not input_list:
        return []

    result = [input_list[0]]
    for item in input_list[1:]:
        if item != result[-1]:
            result.append(item)
    return result


def crop_inner_area(img, region: tuple[int, int]):
    x, w = region
    left = img.crop((0, 0, x, img.height))
    right = img.crop((x + w, 0, img.width, img.height))
    # Create a new image with the combined width
    new_width = left.width + right.width
    new_img = Image.new("RGB", (new_width, img.height))

    # Paste both parts side by side
    new_img.paste(left, (0, 0))
    new_img.paste(right, (left.width, 0))
    return new_img


def read_text(img, region: tuple[int, int, int, int] = None, colored: bool = False, contrast: int = 1, allowlist: str = None, low_text=0.4):
    if region:
        x, y, w, h = region
        img = img.crop((x, y, x + w, y + h))

    if not colored:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if contrast:
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=-(contrast * 50))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    t0 = time.perf_counter()
    raw = reader.predict(img)
    _note_ocr((time.perf_counter() - t0) * 1000.0)
    result = _paddle_texts(raw, allowlist=allowlist, low_text=low_text)
    if config.getboolean('settings', 'debug_mode', fallback=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dev/{timestamp}_{'_'.join(result) if isinstance(result, list) else ''}_{np.random.randint(10, 100):02d}.png"
        cv2.imwrite(filename, img)
    del img
    gc.collect()

    return result


def run_detection_loop(
    state_to_functions: Dict[Optional[str], List[Callable]],
    payload: dict,
):
    while True:
        start_time = time.time()
        try:
            # Capture the screen ONCE per loop
            img, scale_x, scale_y = capture_screen(payload)
            functions = state_to_functions.get(payload.get('state'), [])
            for func in functions:
                if not func:
                    continue
                func(payload, img, scale_x, scale_y)
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Stack trace:")
            print(traceback.format_exc())
        elapsed = time.time() - start_time
        sleep_time = refresh_rate - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


async def send_data(payload, websocket):
    try:
        while True:
            try:
                data = json.dumps(payload)
            except Exception:
                await asyncio.sleep(refresh_rate)
                continue
            size = len(data.encode('utf-8'))
            if size > 1024 * 1024:  # 1MB
                print(f"Warning: Large payload size ({size} bytes)")
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(refresh_rate)
    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        if "no close frame received or sent" not in str(e):
            print(f"Connection error from client: {e}")


async def receive_data(payload: dict, websocket):
    try:
        async for message in websocket:
            if "confirm-entrants:" in message and processing_message is False and config.get('settings', 'capture_mode') == 'game':
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- Received request to confirm players:",
                      str(message).replace("confirm-entrants:", "").strip().split(":"))
                if str(payload['players'][0]['name']) in str(message) and str(payload['players'][1]['name']) in str(message):
                    return True

                def doTask():
                    global processing_message
                    processing_message = True
                    players = str(message).replace(
                        "confirm-entrants:", "").strip().split(":")
                    chosen_player = dialog.choose_player_side(
                        players[0], players[1])
                    if chosen_player == players[0]:
                        payload['players'][0]['name'] = players[0]
                        payload['players'][1]['name'] = players[1]
                    elif chosen_player == players[1]:
                        payload['players'][0]['name'] = players[1]
                        payload['players'][1]['name'] = players[0]
                    processing_message = False
                threading.Thread(target=doTask, daemon=True).start()
                time.sleep(refresh_rate)
    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        if "no close frame received or sent" not in str(e):
            print(f"Connection error from client: {e}")


async def handle_connection(websocket, payload: dict):
    send_task = asyncio.create_task(send_data(payload, websocket))
    receive_task = asyncio.create_task(receive_data(payload, websocket))
    done, pending = await asyncio.wait(
        [send_task, receive_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()


def start_websocket_server(payload: dict):
    import websockets
    import asyncio

    async def start_server(payload: dict):
        async with websockets.serve(
            lambda ws: handle_connection(ws, payload),
            "0.0.0.0",
            config.getint('settings', 'server_port'),
            ping_interval=60,
            ping_timeout=90,
            close_timeout=15
        ):
            await asyncio.Future()  # run forever
    asyncio.run(start_server(payload))


if __name__ == "__main__":
    broadcast_thread = threading.Thread(target=broadcast.broadcast_device_info, args=(
        routines.client_name,), daemon=True).start()
    websocket_thread = threading.Thread(
        target=start_websocket_server, args=(payload,), daemon=True).start()
    print("All systems go. Please head to the character or stage selection screen to start detection.\n")
    run_detection_loop(routines.states_to_functions, payload)
    while True:
        time.sleep(1)
