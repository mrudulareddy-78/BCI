import threading
import queue
import time
import json
import asyncio
import websockets
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

PREDICTION_LOG = "predicted_state_log.txt"

# Shared state
class SharedState:
    def __init__(self):
        self.attention_state = "Neutral"
        self.blink_queue = queue.Queue()
        self.movement_enabled = False

shared = SharedState()

# === EEG attention file watcher ===
class AttentionFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(PREDICTION_LOG):
            try:
                with open(PREDICTION_LOG, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if lines:
                        latest = lines[-1].strip()
                        if "→" in latest:
                            _, state = latest.split("→")
                            shared.attention_state = state.strip()
                            print(f"[EEG] Updated state: {shared.attention_state}")
            except Exception as e:
                print(f"[EEG] Error reading log: {e}")

def start_eeg_watcher():
    handler = AttentionFileHandler()
    observer = Observer()
    observer.schedule(handler, ".", recursive=False)
    observer.start()

# === Movement decision logic ===
def interpret_blink(blink_type):
    if blink_type == "long":
        shared.movement_enabled = not shared.movement_enabled
        print(f"[SYSTEM] Movement {'enabled' if shared.movement_enabled else 'disabled'}")

    elif shared.movement_enabled and shared.attention_state in ["Focused", "Neutral"]:
        if blink_type == "double":
            return {"command": "MOVE", "direction": "RIGHT"}
        elif blink_type == "triple":
            return {"command": "MOVE", "direction": "LEFT"}
    return None

# === WebSocket sender ===
async def send_movements(uri):
    async with websockets.connect(uri) as ws:
        print("[WebSocket] Connected to Unity")

        while True:
            blink_type = shared.blink_queue.get()
            print(f"[BLINK] Received: {blink_type}")

            movement = interpret_blink(blink_type)
            if movement:
                await ws.send(json.dumps(movement))
                print(f"[ACTION] Sent to Unity: {movement}")

# === Thread for WebSocket + blink queue ===
def start_websocket_client():
    asyncio.run(send_movements("ws://localhost:8765"))  # change port if needed

# === Main startup ===
if __name__ == "__main__":
    print("[CONTROL] Starting hybrid control logic...")
    start_eeg_watcher()

    # Start WebSocket movement thread
    ws_thread = threading.Thread(target=start_websocket_client, daemon=True)
    ws_thread.start()

    # Import and run blink detector
    from blink_detector import EyeBlinkDetector
    blink_detector = EyeBlinkDetector(shared.blink_queue)
    blink_detector.start()

    blink_detector.join()
