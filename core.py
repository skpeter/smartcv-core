import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
if __name__ == "__main__":
    print("Initializing...")
    from routines import client_name
    try:
        from build_info import __version__
    except: __version__ = "DEV"
    print(f"Welcome to SmartCV type: {client_name.split("-")[1]} - build: {__version__}")
    from routines import payload
import configparser
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from typing import Callable, Dict, List, Optional
import routines
import dialog
import broadcast
import cv2
import numpy as np
import threading
import time
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
config = configparser.ConfigParser()
config.read('config.ini')
processing_message = False
reader = easyocr.Reader(['en'])

refresh_rate = config.getfloat('settings', 'refresh_rate')
capture_mode = config.get('settings', 'capture_mode')
executable_title = config.get('settings', 'executable_title', fallback="")
feed_path = config.get('settings', 'feed_path')

base_height = 1080
base_width = 1920



def print_with_time(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp, "-", *args, **kwargs)

# Check if the pixel color is within the deviation range
def is_within_deviation(color1, color2, deviation):
    return all(abs(c1 - c2) / 255.0 <= deviation for c1, c2 in zip(color1, color2))

def capture_screen():
    if capture_mode == 'obs':
        while True:
            try:
                img = Image.open(feed_path)
                break
            except (OSError, Image.UnidentifiedImageError) as e:
                if "truncated" in str(e) or "cannot identify image file" in str(e) or "could not create decoder object" in str(e):
                    # print("Image is truncated or cannot be identified. Retrying...")
                    time.sleep(0.1)
                    continue
                else:
                    raise e
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

def detect_image(img, template_file:str, region:tuple[int, int, int, int]=None):
    # Crop the specific area
    if region:
        x, y, w, h = region
        img = img.crop((x, y, x + w, y + h))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Load the template images
    template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
    
    if template is None:
        raise FileNotFoundError("Template image not found")
    
    # Perform template matching
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    return np.max(res)

def crop_image_y(img, crop_area:tuple[int,int]):
    crop_start, crop_end = crop_area
    img = np.array(img)

    # Remove the single text area from the image
    left = img[:, :crop_start]
    right = img[:, crop_end:]
    return np.hstack([left, right])

def get_color_match_in_region(img, region:tuple[int, int, int, int], target_color:tuple, deviation:float):
    x, y, w, h = region
    cropped_area = img.crop((x, y, x + w, y + h))
    deviation = 0.15
    width, height = cropped_area.size
    total_pixels = width * height
    matching_pixels = 0

    for i in range(width):
        for j in range(height):
            if is_within_deviation(cropped_area.getpixel((i, j)), target_color, deviation):
                matching_pixels += 1
    return matching_pixels / total_pixels

def remove_neighbor_duplicates(input_list):
    if not input_list:
        return []

    result = [input_list[0]]
    for item in input_list[1:]:
        if item != result[-1]:
            result.append(item)
    return result

def read_text(img, region: tuple[int, int, int, int]=None, colored:bool=False, contrast:int=None, allowlist:str=None, low_text=0.4, contrast_ths=0.7):
    # print("Attempting to read text...")
    # Define the area to read
    if region:
        x, y, w, h = region
        img = img.crop((x, y, x + w, y + h))

    if not colored: img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    else: img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if contrast: img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
    result = reader.readtext(img, paragraph=False, allowlist=allowlist, contrast_ths=contrast_ths, low_text=low_text)
    if config.getboolean('settings', 'debug_mode', fallback=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cropped_{timestamp}.png"
        cv2.imwrite(filename, img)

    # Extract the text
    if result:
        result = ' '.join([res[1] for res in result])
    else: result = None

    # Release memory
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


def run_detection_loop(
    state_to_functions: Dict[Optional[str], List[Callable]],
    payload: dict,
):
    while True:
        threads = []
        try:
            functions = state_to_functions.get(payload.get('state'), [])
            for func in functions:
                if not func: continue
                t = threading.Thread(target=func, args=(payload,))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Stack trace:")
            print(traceback.format_exc())
        time.sleep(config.getfloat('settings', 'refresh_rate'))

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
    print("All systems go. Please head to the character selection screen to start detection.\n")

    while True:
        time.sleep(1)