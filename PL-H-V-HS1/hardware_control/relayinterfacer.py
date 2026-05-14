import ctypes
import time
import os
import config
import sys
import subprocess
import threading
import cv2
from datetime import datetime

# --- Paths & Config ---
RELAY_DLL_PATH   = r"C:\Users\Admin\Downloads\USBB-RELAY(Softwares)\USBB-RELAY(Softwares)\usb_relay_device.dll"
BOSON_SDK_DIR    = r"C:\Program Files (x86)\FLIR Systems\BosonApp"
BOSON_COM_PORT   = "COM4"
BOSON_BAUD_RATE  = 921600
BOSON_CAM_INDEX  = 0

PRE_RELAY_SECS   = 1
RELAY_PULSE_SECS = 1
POST_RELAY_SECS  = 10

CAPTURES_ROOT    = config.BOSON_ROOT + r""
ANALYSIS_SCRIPT  = os.path.join(os.path.dirname(__file__), "heatflux_analysis.py")

# 64-bit Python for analysis (scipy/matplotlib need it)
PYTHON64 = sys.executable.replace("Python312-32", "Python312") \
           if "Python312-32" in sys.executable else "python"


# --- Boson SDK init ---
def init_boson_sdk():
    sys.path.insert(0, BOSON_SDK_DIR)
    import clr
    clr.AddReference(os.path.join(BOSON_SDK_DIR, "BosonSDK"))
    from Boson import Camera
    cam = Camera()
    cam.Initialize(BOSON_COM_PORT, BOSON_BAUD_RATE)
    sn = None
    cam.sysinfoGetCameraSN(sn)
    print(f"Boson connected on {BOSON_COM_PORT}")
    return cam


# --- Relay init ---
def init_relay():
    if not os.path.exists(RELAY_DLL_PATH):
        raise FileNotFoundError(f"Relay DLL not found: {RELAY_DLL_PATH}")
    lib = ctypes.CDLL(RELAY_DLL_PATH)
    lib.usb_relay_init.restype                           = ctypes.c_int
    lib.usb_relay_device_enumerate.restype               = ctypes.c_void_p
    lib.usb_relay_device_open.argtypes                   = [ctypes.c_void_p]
    lib.usb_relay_device_open.restype                    = ctypes.c_void_p
    lib.usb_relay_device_open_one_relay_channel.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.usb_relay_device_close_one_relay_channel.argtypes= [ctypes.c_void_p, ctypes.c_int]
    lib.usb_relay_device_close.argtypes                  = [ctypes.c_void_p]
    lib.usb_relay_exit.restype                           = ctypes.c_int
    if lib.usb_relay_init() != 0:
        raise RuntimeError("Relay library init failed")
    device_info = lib.usb_relay_device_enumerate()
    if not device_info:
        lib.usb_relay_exit()
        raise RuntimeError("No relay board detected — check USB connection")
    handle = lib.usb_relay_device_open(device_info)
    if not handle:
        lib.usb_relay_exit()
        raise RuntimeError("Failed to open relay device handle")
    print("Relay ready")
    return lib, handle

def close_relay(lib, handle):
    lib.usb_relay_device_close(handle)
    lib.usb_relay_exit()


# --- Video capture thread ---
stop_capture = threading.Event()

def capture_frames(cap, tiff_dir):
    """Save each frame as a 16-bit TIFF preserving full radiometric data."""
    os.makedirs(tiff_dir, exist_ok=True)
    frame_count = 0
    consecutive_failures = 0
    MAX_FAILURES = 30
    while not stop_capture.is_set():
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= MAX_FAILURES:
                print(f"Camera stream lost after {MAX_FAILURES} consecutive failures — stopping capture.")
                break
            time.sleep(0.05)
            continue
        consecutive_failures = 0
        # Extract single-channel raw data — preserve 16-bit if available
        if len(frame.shape) == 3:
            raw = frame[:, :, 0]   # Y channel from Y16 UVC stream
        else:
            raw = frame
        tiff_path = os.path.join(tiff_dir, f"frame_{frame_count:05d}.tiff")
        cv2.imwrite(tiff_path, raw)
        frame_count += 1
    print(f"Capture complete: {frame_count} frames -> {tiff_dir}")
    return frame_count


# --- Analysis launcher ---
def run_analysis(video_path, session_dir):
    print(f"\nLaunching analysis ({PYTHON64}) ...")
    result = subprocess.run(
        [PYTHON64, ANALYSIS_SCRIPT, video_path, session_dir],
        capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"Analysis error:\n{result.stderr}")
    else:
        print("Analysis complete.")


# --- Main sequence ---
def run():
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(CAPTURES_ROOT, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    tiff_dir    = os.path.join(session_dir, f"boson_{timestamp}")

    print(f"\n{'='*55}")
    print(f"Session: {session_dir}")
    print(f"{'='*55}")

    # Open UVC video stream FIRST — before SDK touches the camera over serial,
    # otherwise the SDK's Initialize call resets the camera and drops the stream.
    cap = cv2.VideoCapture(BOSON_CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open Boson video stream (index {BOSON_CAM_INDEX})")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Boson stream: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {fps:.1f} fps")

    cam = init_boson_sdk()
    relay_lib, relay_handle = init_relay()

    try:
        stop_capture.clear()
        capture_thread = threading.Thread(
            target=capture_frames, args=(cap, tiff_dir), daemon=True
        )
        capture_thread.start()
        print(f"[{datetime.now():%H:%M:%S}] Capture started — "
              f"waiting {PRE_RELAY_SECS}s before relay fires...")
        time.sleep(PRE_RELAY_SECS)

        print(f"[{datetime.now():%H:%M:%S}] Relay 1: ON")
        relay_lib.usb_relay_device_open_one_relay_channel(relay_handle, 1)
        time.sleep(RELAY_PULSE_SECS)
        print(f"[{datetime.now():%H:%M:%S}] Relay 1: OFF")
        relay_lib.usb_relay_device_close_one_relay_channel(relay_handle, 1)

        print(f"[{datetime.now():%H:%M:%S}] Capturing {POST_RELAY_SECS}s post-relay...")
        time.sleep(POST_RELAY_SECS)

    finally:
        stop_capture.set()
        capture_thread.join(timeout=5)
        cap.release()
        close_relay(relay_lib, relay_handle)
        cam.Close()

    # Launch analysis in 64-bit Python
    run_analysis(tiff_dir, session_dir)

    print(f"\nSession folder: {session_dir}")


if __name__ == "__main__":
    run()
