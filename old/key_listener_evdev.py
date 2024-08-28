import socket
import json
import argparse
import time
import sys
import evdev
from evdev import UInput, ecodes as e
from select import select

parser = argparse.ArgumentParser(description="Run the key listener for activation engineering")
parser.add_argument("--host", type=str, default=socket.gethostname(), help="Host for the connector")
parser.add_argument("--port", type=int, default=5005, help="Port for the connector")
parser.add_argument("--toggling-key", type=str, default="KEY_LEFTMETA", help="Key used for toggling")
args = parser.parse_args()

host = args.host
port = args.port

client_socket = socket.socket()

print("Waiting for connection", end="")
while True:
    try:
        client_socket.connect((host, port))
        break
    except ConnectionRefusedError:
        print(".", end="")
        sys.stdout.flush()
        time.sleep(1)
        continue
print("\nConnected")

class KeyHandler:
    def __init__(self, toggling_key=args.toggling_key):
        self._toggling_key = toggling_key
        self.pressed = set()
        self._toggle_key_pressed_alone = False
        self.generating_lock = False
        
        # Find all keyboard devices
        self.devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
        self.keyboards = []
        for device in self.devices:
            caps = device.capabilities()
            if e.EV_KEY in caps:
                key_caps = caps[e.EV_KEY]
                if e.ecodes[self._toggling_key] in key_caps:
                    self.keyboards.append(device)
        
        if not self.keyboards:
            raise Exception(f"No keyboard found with the specified toggling key: {self._toggling_key}")
        
        print(f"Found {len(self.keyboards)} suitable keyboard(s)")
        for kb in self.keyboards:
            print(f"  - {kb.name}")

        self.writer = UInput()

    def get_state(self):
        return dict(
            pressed=sorted(self.pressed),
            generating_lock=self.generating_lock,
            should_generate=self.should_generate(),
        )

    def send_state(self, state):
        print(state)
        # return
        try:
            client_socket.sendall(("\n" + json.dumps(state)).encode())
        except BrokenPipeError:
            print("Server disconnected")
            sys.exit(0)

    def handle_event(self, event):
        if event.type == e.EV_KEY:
            last_state = self.get_state()
            key = e.KEY[event.code]
            key_name = key.replace("KEY_", "").lower()
            
            if event.value == 1:  # Key press
                self.pressed.add(key_name)
                if key == self._toggling_key:
                    self._toggle_key_pressed_alone = True
                else:
                    self._toggle_key_pressed_alone = False
            elif event.value == 0:  # Key release
                if key in self.pressed:
                    self.pressed.remove(key_name)
                if key == self._toggling_key and self._toggle_key_pressed_alone:
                    self.generating_lock = not self.generating_lock
                    print(f"Generating lock: {'ON' if self.generating_lock else 'OFF'}")

            new_state = self.get_state()
            if new_state != last_state:
                self.send_state(new_state)

    def should_generate(self):
        if self.generating_lock:
            return True
        if "KEY_LEFTALT" in self.pressed and "KEY_LEFTSHIFT" in self.pressed:
            return True
        return False

    def run(self):
        try:
            while True:
                r, w, x = select(self.keyboards, [], [])
                for fd in r:
                    for event in fd.read():
                        self.handle_event(event)
        except KeyboardInterrupt:
            self.cleanup()

    def cleanup(self):
        final_state = dict(
            pressed=[],
            generating_lock=False,
            should_generate=False,
        )
        self.send_state(final_state)
        self.writer.close()
        for kb in self.keyboards:
            kb.close()
        client_socket.close()
        print("\nBye")

if __name__ == "__main__":
    try:
        key_handler = KeyHandler()
        key_handler.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)