import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import routines
import configparser
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from typing import Callable, Dict, List, Optional
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
from datetime import datetime
config = configparser.ConfigParser()
config.read('config.ini')
previous_states = [None] # list of previous states to be used for state change detection
processing_message = False
lock = threading.Lock()

reader = easyocr.Reader(['en'])

refresh_rate = config.getfloat('settings', 'refresh_rate')
capture_mode = config.get('settings', 'capture_mode')
executable_title = config.get('settings', 'executable_title')
feed_path = config.get('settings', 'feed_path')

base_height = 1080
base_width = 1920


payload = {
    "state": None,
    "players": [
        {
            "name": None,
            "character": None,
            "rounds": 2,
        },
        {
            "name": None,
            "character": None,
            "rounds": 2,
        }
    ]
}

def print_with_timestamp(message):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),f"- {message}")

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
    else:
        # Find the window by its title
        windows = gw.getWindowsWithTitle(executable_title)
        if windows:
            window = windows[0]
        else:
            print(f"Executable {executable_title} not found. Ensure it is running and visible.")
            return False, None, None

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

def get_color_match_in_region(img: ImageFile, region:tuple[int, int, int, int], target_color:tuple, deviation:float):
    cropped_area = img.crop(region)
    deviation = 0.15
    width, height = cropped_area.size
    total_pixels = width * height
    matching_pixels = 0

    for i in range(width):
        for j in range(height):
            if is_within_deviation(cropped_area.getpixel((i, j)), target_color, deviation):
                matching_pixels += 1
    return matching_pixels / total_pixels

def read_text(img, region: tuple[int, int, int, int], colored:bool=False, contrast:int=None):
    # print("Attempting to read text...")
    # Define the area to read
    x, y, w, h = region
    cropped_img = img.crop((x, y, x + w, y + h))
    cropped_img = np.array(cropped_img)

    if not colored: cropped_img = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2GRAY)
    else: cropped_img = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
    if contrast: cropped_img = cv2.convertScaleAbs(cropped_img, alpha=contrast, beta=0)
    # Use OCR to read the text from the image
    result = reader.readtext(cropped_img, paragraph=False)

    # Extract the text
    if result:
        result = ' '.join([res[1] for res in result])
    else: result = None

    # Release memory
    del cropped_img
    gc.collect()

    return result

def run_detection_loop(
    state_to_functions: Dict[Optional[str], List[Callable]],
    payload: dict,
    lock: threading.Lock,
):
    while True:
        threads = []
        try:
            functions = state_to_functions.get(payload.get('state'), [])
            for func in functions:
                t = threading.Thread(target=func, args=(payload, lock))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Stack trace:")
            print(traceback.format_exc())
        time.sleep(config.get('settings', 'refresh_rate'))

async def send_data(payload, websocket):
    try:
        while True:
            data = json.dumps(payload)
            size = len(data.encode('utf-8'))
            if size > 1024 * 1024:  # 1MB
                print(f"Warning: Large payload size ({size} bytes)")
            refresh_rate = config.getfloat('settings', 'refresh_rate')
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(refresh_rate)
    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        if "no close frame received or sent" not in str(e):
            print(f"Connection error from client: {e}")

async def receive_data(payload:dict, websocket, lock: threading.Lock):
    try:
        async for message in websocket:
            if "confirm-entrants:" in message and processing_message == False: # and config.get('settings', 'capture_mode') == 'game':
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),f"- Received request to confirm players:", str(message).replace("confirm-entrants:", "").strip().split(":"))
                if str(payload['players'][0]['name']) in str(message) and str(payload['players'][1]['name']) in str(message): return True
                def doTask():
                    with lock:
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

async def handle_connection(payload:dict, websocket, lock: threading.Lock):
    send_task = asyncio.create_task(send_data(payload, websocket))
    receive_task = asyncio.create_task(receive_data(payload, websocket, lock))
    done, pending = await asyncio.wait(
        [send_task, receive_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

def start_websocket_server(payload:dict, lock: threading.Lock):
    import websockets
    import asyncio

    async def start_server(payload:dict, lock: threading.Lock):
        async with websockets.serve(
            lambda ws: handle_connection(ws, payload, lock),
            "0.0.0.0",
            config.getint('settings', 'server_port'),
            ping_interval=60,
            ping_timeout=90,
            close_timeout=15
        ):
            await asyncio.Future()  # run forever
    asyncio.run(start_server(payload, lock))

if __name__ == "__main__":
    print("Initializing...")
    broadcast_thread = threading.Thread(target=broadcast.broadcast_device_info, args=(routines.client_name,), daemon=True).start()
    detection_thread = threading.Thread(target=run_detection_loop, args=(routines.states_to_functions, lock), daemon=True).start()
    websocket_thread = threading.Thread(target=start_websocket_server, args=(payload, lock), daemon=True).start()

    time.sleep(1)
    print("All systems go. Please head to the character selection screen to start detection.\n")

    while True:
        time.sleep(1)