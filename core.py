import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
if __name__ == "__main__":
    print("Initializing...")
    from routines import client_name
    try:
        from build_info import __version__ # type: ignore
    except: __version__ = "DEV"
    print(f"Welcome to {client_name.upper()} - build: {__version__}")
    from routines import payload
import time
import configparser
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from typing import Callable, Dict, List, Optional
import routines
from core import dialog
from core import broadcast
import cv2
import numpy as np
import threading
import easyocr
import gc
import json
import websockets
import asyncio
import mss
import pygetwindow as gw
import traceback
import requests
from datetime import datetime
import obsws_python as obsws
import base64
from io import BytesIO

config = configparser.ConfigParser()
config.read('config.ini')
processing_message = False
if 'reader' not in globals():
    reader = easyocr.Reader(['en'])
refresh_rate = config.getfloat('settings', 'refresh_rate')
capture_mode = config.get('settings', 'capture_mode')
executable_title = config.get('settings', 'executable_title', fallback="")
obs = None
base_height = 1080
base_width = 1920

def print_with_time(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp, "-", *args, **kwargs)

def capture_screen(payload):
    global obs
    if capture_mode == 'obs':
        try: 
            if not obs: obs = obsws.ReqClient(
                host=config.get('obs', 'host', fallback='localhost'),
                port=config.get('obs', 'port', fallback=4455),
                password=config.get('obs', 'password', fallback='')
            )
        except Exception as e:
            print_with_time("Could not connect to OBS. Retrying...")
            payload['state'] = None
            return None, 1.0, 1.0
        while True:
            response = obs.get_source_screenshot(
                name=config.get('obs', 'source_title', fallback=""),
                img_format="webp",
                width=config.getint('obs', 'width', fallback=1920),
                height=config.getint('obs', 'height', fallback=1080),
                quality=95
            )
            prefix = "base64,"
            idx = response.image_data.find(prefix)
            img_str = response.image_data[idx + len(prefix):] if idx != -1 else response.image_data
            img_data = base64.b64decode(img_str)
            img = Image.open(BytesIO(img_data))
            break
    elif capture_mode == 'game':
        capture_attempts = 0
        while True:
            windows = gw.getWindowsWithTitle(executable_title)
            if windows:
                window = windows[0]
                if capture_attempts > 0: print(f"Found executable {executable_title}")
                break
            else:
                if capture_attempts < 1: print(f"Executable {executable_title} not found. Ensure it is running and visible.")
                capture_attempts += 1
                payload['state'] = None
                continue

        # Get the window's bounding box
        # Get the window's dimensions
        width = window.right - window.left
        height = window.bottom - window.top
        
        # Calculate target height for 16:9 aspect ratio
        target_height = int(width * (9/16))
        
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
    return np.all(np.abs(np.array(pixel[:3] if type(pixel) == tuple else [pixel, pixel, pixel]) - np.array(target_color)) <= 255 * deviation)

def find_color_runs_np(row, color, deviation=0.1):
    runs = []
    in_run = False
    start_x = 0
    for x, pixel in enumerate(row):
        if is_within_deviation(pixel, color, deviation):
            if not in_run:
                in_run = True
                start_x = x
        else:
            if in_run:
                in_run = False
                runs.append((start_x, x - 1))
    if in_run:
        runs.append((start_x, len(row) - 1))
    return runs

def merge_runs_with_margin(runs, margin, width):
    merged = []
    for start, end in runs:
        start = max(start - margin, 0)
        end = min(end + margin, width - 1)
        if not merged:
            merged.append((start, end))
        else:
            last_start, last_end = merged[-1]
            if start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
    return merged

def stitch_text_regions(image_array, y_line, color, margin=10, deviation=0.1):
    if image_array.ndim == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    bgr_image = image_array

    row = bgr_image[y_line]
    raw_runs = find_color_runs_np(row, color, deviation)
    if not raw_runs:
        return np.empty((0, 0, 0))

    width = image_array.shape[1]
    merged_runs = merge_runs_with_margin(raw_runs, margin, width)

    cropped_strips = []
    for start_x, end_x in merged_runs:
        strip = image_array[:, start_x:end_x + 1]
        cropped_strips.append(strip)

    total_width = sum(strip.shape[1] for strip in cropped_strips)
    stitched = np.zeros((image_array.shape[0], total_width, image_array.shape[2] if len(image_array.shape) == 3 else image_array.shape[0]), dtype=image_array.dtype)

    x_offset = 0
    for strip in cropped_strips:
        stitched[:, x_offset:x_offset + strip.shape[1]] = strip
        x_offset += strip.shape[1]

    return stitched

def resize_template(template, scale_x, scale_y):
    h, w = template.shape[:2]
    return cv2.resize(template, (int(w * scale_x), int(h * scale_y)), interpolation=cv2.INTER_AREA)

def detect_image(img, scale_x, scale_y, template_file:str, region:tuple[int, int, int, int]=None):
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

def get_color_match_in_region(img, region:tuple[int, int, int, int], target_color:tuple|list[tuple], deviation:float=0.15):
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

def read_text(img, region: tuple[int, int, int, int]=None, colored:bool=False, contrast:int=1, allowlist:str=None, low_text=0.4):
    if region:
        x, y, w, h = region
        img = img.crop((x, y, x + w, y + h))

    if not colored: img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    else: img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if contrast: img = cv2.convertScaleAbs(img, alpha=contrast, beta=-(contrast * 50))

    result = reader.readtext(img, paragraph=False, allowlist=allowlist, low_text=low_text)

    if result:
        result = [res[1] for res in result]
    else: result = None
    if config.getboolean('settings', 'debug_mode', fallback=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dev/{'_'.join(result) if isinstance(result, list) else ''}_{timestamp}.png"
        cv2.imwrite(filename, img)
    del img
    gc.collect()

    return result

def get_latest_build_number():
    url = f'https://api.github.com/repos/skpeter/{client_name}/releases/latest'
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        tag = r.json()['tag_name']
        if 'release-main-' in tag:
            return int(tag.rsplit('-', 1)[-1])
    except Exception as e:
        print(f"Update check failed: {e}")
    return None

def is_update_available():
    latest = get_latest_build_number()
    return latest if latest is not None and int(__version__) < latest else False

last_mtime = None
def run_detection_loop(
    state_to_functions: Dict[Optional[str], List[Callable]],
    payload: dict,
):
    while True:
        threads = []
        start_time = time.time()
        try:
            # Capture the screen ONCE per loop
            img, scale_x, scale_y = capture_screen(payload)
            functions = state_to_functions.get(payload.get('state'), [])
            for func in functions:
                if not func: continue
                # Pass the image and scales to each function
                t = threading.Thread(target=func, args=(payload, img, scale_x, scale_y))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
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
            except Exception as e:
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

async def receive_data(payload:dict, websocket):
    try:
        async for message in websocket:
            if "confirm-entrants:" in message and processing_message == False and config.get('settings', 'capture_mode') == 'game':
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),f"- Received request to confirm players:", str(message).replace("confirm-entrants:", "").strip().split(":"))
                if str(payload['players'][0]['name']) in str(message) and str(payload['players'][1]['name']) in str(message): return True
                def doTask():
                    global processing_message
                    processing_message = True
                    players = str(message).replace("confirm-entrants:", "").strip().split(":")
                    chosen_player = dialog.choose_player_side(players[0], players[1])
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

async def handle_connection(websocket, payload:dict):
    send_task = asyncio.create_task(send_data(payload, websocket))
    receive_task = asyncio.create_task(receive_data(payload, websocket))
    done, pending = await asyncio.wait(
        [send_task, receive_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

def start_websocket_server(payload:dict):
    import websockets
    import asyncio

    async def start_server(payload:dict):
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
    new_ver = is_update_available()
    if new_ver: print(f"New build {new_ver} available (you are on build {__version__}). Head over to \nhttps://github.com/skpeter/{client_name} to download it.")
    broadcast_thread = threading.Thread(target=broadcast.broadcast_device_info, args=(routines.client_name,), daemon=True).start()
    detection_thread = threading.Thread(target=run_detection_loop, args=(routines.states_to_functions, payload), daemon=True).start()
    websocket_thread = threading.Thread(target=start_websocket_server, args=(payload,), daemon=True).start()
    print("All systems go. Please head to the character or stage selection screen to start detection.\n")
    while True:
        time.sleep(1)