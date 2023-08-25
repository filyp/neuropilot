# %%
import socket
import json
import argparse

parser = argparse.ArgumentParser(description='Run the key listener for activation engineering')
parser.add_argument('--connector_port', type=int, default=5005, help='Port for the connector')
args = parser.parse_args()

host = socket.gethostname()
port = args.connector_port

client_socket = socket.socket()
client_socket.connect((host, port))

# %%
from pynput import keyboard
# "keyboard" lib would let us access what is pressed directly, but it requires root


class KeyHandler:
    def __init__(self, toggling_key="cmd"):
        # toggling_key is the key that will be used to toggle the generating lock
        # good candidates are "cmd", "alt", "ctrl"
        self._toggling_key = toggling_key
        self.pressed = set()
        # self.esc_registered = False
        self._toggle_key_pressed_alone = False
        self.generating_lock = False
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
    
    def get_state(self):
        return dict(
            pressed=sorted(self.pressed),
            # esc_registered=self.esc_registered,
            generating_lock=self.generating_lock,
            should_generate=self.should_generate(),
        )
    
    def send_state(self, state):
        client_socket.sendall(("\n" + json.dumps(state)).encode())

    def _on_press(self, key):
        last_state = self.get_state()

        k = str(key).replace("'", "").replace("Key.", "").replace("<65511>", "alt").lower()
        if k == "<0>":
            return    # this is some weird macro artifact
        # if k == "esc":
        #     # self.esc_registered = True
        #     # close connection
        #     client_socket.close()
        #     # stop the listener
        #     return False
        # print(f"key {k} pressed") 
        self.pressed.add(k)
        
        # implement toggling behavior
        if k == self._toggling_key and len(self.pressed) == 1:
            self._toggle_key_pressed_alone = True
        if k != self._toggling_key:
            self._toggle_key_pressed_alone = False
        
        new_state = self.get_state()
        if new_state != last_state:
            # state changed, so send it to the server
            self.send_state(new_state)

    def _on_release(self, key): 
        last_state = self.get_state()

        k = str(key).replace("'", "").replace("Key.", "").replace("<65511>", "alt").lower()
        # print(f"key {k} released") 
        if k in self.pressed:
            self.pressed.remove(k)

        # implement toggling behavior
        if k == self._toggling_key and self._toggle_key_pressed_alone:
            # toggle key was tapped w/o anything else
            self.generating_lock = not self.generating_lock
            # it's unclean to reference ui here, but it's the easiest way
            # I coulc also use a callback
            # ui.text_area.disabled = self._generating_lock
        
        new_state = self.get_state()
        if new_state != last_state:
            # state changed, so send it to the server
            self.send_state(new_state)

    def should_generate(self):
        # if self.esc_registered:
        #     # just to be sure esc can always stop; maybe not needed
        #     return False

        if self.generating_lock:
            # if generating lock is on, we don't want to stop
            return True
        if "alt" in self.pressed and "shift" in self.pressed:
            return True

        # no reason to continue generating
        return False
                

try:
    key_handler = KeyHandler()
    key_handler.listener.join()
except KeyboardInterrupt:
    key_handler.listener.stop()

    final_state = dict(
        pressed=[],
        generating_lock=False,
        should_generate=False,
    )
    key_handler.send_state(final_state)
    client_socket.close()
    print("\nbye")
    
