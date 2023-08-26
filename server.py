# original activation engineering code: https://colab.research.google.com/drive/1y84fhgkGX0ft2DmYJB3K13lAyf-0YonK?usp=sharing#scrollTo=ZExJFurIjKHM
# some supported models: https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=jHj79Pj58cgJKdq4t-ygK-4h
import time
import socket
import json
import argparse
from threading import Thread
from typing import Dict, Union, List, Tuple

import hashlib
import matplotlib
import matplotlib.pyplot as plt
import panel as pn
import torch
import numpy as np
from transformer_lens import HookedTransformer


parser = argparse.ArgumentParser(description='Run the server for activation engineering')
parser.add_argument('--ui_port', type=int, default=5000, help='Port for the UI')
parser.add_argument('--connector_port', type=int, default=5005, help='Port for the connector')
parser.add_argument('--model_name', type=str, default="gpt2-small", help='Name of the model to use')
args = parser.parse_args()


class ClientConnector:
    def __init__(self):
        self.pressed = set()
        self.generating_lock = False
        self.should_generate = False

        host = socket.gethostname()
        port = args.connector_port

        self._server_socket = socket.socket()
        self._server_socket.bind((host, port))
        self._server_socket.settimeout(3)
        
        self.stop = False
        self.receiver_thread = Thread(target=self.receive_data)
        # self.receiver_thread.start()
    
    def receive_data(self):
        self._server_socket.listen(1)
        
        while not self.stop:
            try:
                conn, address = self._server_socket.accept()
            except socket.timeout:
                # print("Waiting for connection...")
                continue
            print("Connection from: " + str(address))

            while not self.stop:
                data = conn.recv(1024).decode()
                if not data:
                    break
                
                # the data can be two dicts concatenated,
                # so we need to split them and only process the last one
                data = data.split("\n")[-1]
                data_dict = json.loads(data)

                self.pressed = set(data_dict["pressed"])
                self.generating_lock = data_dict["generating_lock"]
                self.should_generate = data_dict["should_generate"]
                # note: for some reason "/" press isn't processed right away, but something else needs to "flush" it
                # so it cannot be used now for generating
                # probably it's because of weird escaping

            conn.close()
            print("Connection closed")
        print("Connector thread stopped")


class UI:
    def __init__(self):
        self.text_area = pn.widgets.TextAreaInput(value="", sizing_mode="stretch_both")
        self.steering_strength = pn.widgets.FloatSlider(name="Steering Strength (log10)", start=-3, end=3, step=0.01, value=0)
        self.layer_num = pn.widgets.IntSlider(name="Layer Number", start=0, end=num_layers - 1, step=1, value=6)
        self.info_box = pn.widgets.StaticText(name="Info", value="Not started")

        # some square for 2d plotting, must have square aspect ratio
        # turn interactive plotting off, so that it's not displayed in notebook
        plt.ioff()
        fig, ax = plt.subplots(figsize=(20, 20))

        ax.set_aspect("equal") # note: not sure if clear resets this or not
        ax.set_facecolor((0., 0., 0.))  # black background
        self._ax = ax
        self._max_plotting_scale = 0.000001
        self.update_plot(None, None)  # set up plot
        self.plot = pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_both", format="svg")
        # svg format is necessary; without it there are some weird lags when updating the text!
        
        self.full = pn.Row(
            self.text_area,
            pn.Column(
                pn.Row(self.steering_strength, self.layer_num, sizing_mode="stretch_width"),
                self.info_box,
                self.plot,
                sizing_mode="stretch_both"
            ),
        )
    
    def update_plot(self, existing_activation: List[float], modifying_activations: List[Tuple[List[float], str]]):
        ax = self._ax
        # clear previous plot
        ax.clear()
        # plot formatting
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title("Click to start")
        # draw axis lines
        ax.plot([-1, 1], [0, 0], color="grey", linewidth=1)
        ax.plot([0, 0], [-1, 1], color="grey", linewidth=1)
        if existing_activation is None:
            return
        
        # update scale
        vector_sum = np.array(existing_activation[:2])
        self._max_plotting_scale = max(np.abs(vector_sum[0]), np.abs(vector_sum[1]), self._max_plotting_scale)
        for activation, _ in modifying_activations:
            vector_sum += np.array(activation[:2])
            self._max_plotting_scale = max(np.abs(vector_sum[0]), np.abs(vector_sum[1]), self._max_plotting_scale)
        s = self._max_plotting_scale

        # draw existing activation
        vector_sum = np.array(existing_activation[:2])
        # ax.arrow(0, 0, vector_sum[0] / s, vector_sum[1] / s, color="white", linewidth=4, head_width=0.04, head_length=0.04)
        ax.plot([0, vector_sum[0] / s], [0, vector_sum[1] / s], color="white", linewidth=4)
        # draw modifying activations
        for activation, key in modifying_activations:
            # convert key (string) to color by hashing
            hue = int(hashlib.shake_128(key.encode('utf-8')).hexdigest(1), 16)
            color = matplotlib.colors.hsv_to_rgb((hue / 255, 1, 1))
            
            # ax.arrow(vector_sum[0] / s, vector_sum[1] / s, activation[0] / s, activation[1] / s, color="red", linewidth=2, head_width=0.02, head_length=0.02)
            new_vector_sum = vector_sum + np.array(activation[:2])
            ax.plot(
                [vector_sum[0] / s, new_vector_sum[0] / s],
                [vector_sum[1] / s, new_vector_sum[1] / s],
                color=color,
                linewidth=2
            )
            vector_sum = new_vector_sum


        # update plot
        self.plot.param.trigger('object')


# load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)  # save memory
model = HookedTransformer.from_pretrained(args.model_name, device=device)
num_layers = len(model._modules["blocks"])

client_connector = ClientConnector()
ui = UI()

# consctruct a random set of directions
np.random.seed(0)
directions = dict()
for letter in "abcdefghijklmnopqrstuvwxyz,.":
    directions[letter] = np.random.normal(0, 1, model.cfg.d_model)
# TODO handling these directions could be done by a class, together with looping over pressed keys, and later extracting directions from text
    

def add_vector(resid_pre, hook):
    if hook.layer() != ui.layer_num.value:
        return

    to_add = np.zeros(model.cfg.d_model)
    modifying_activations = []
    for key in client_connector.pressed:
        if key in directions:
            component = directions[key] * (10 ** ui.steering_strength.value)
            to_add += component
            modifying_activations.append((component[:2], key))
    
    ui.update_plot(resid_pre[:, -1, :2].flatten().cpu(), modifying_activations)

    resid_pre[:, -1, :] += torch.from_numpy(to_add).cuda()
    # TODO double check that this broadcasting works as intended
    

def new_token_callback(tokens, hooked_transformer):
    tokens_to_display = tokens[0][1:]   # remove the BOS token
    text = hooked_transformer.tokenizer.decode(tokens_to_display)
    ui.text_area.value_input = text
    # btw, update all the rest
    ui.info_box.value = str(client_connector.pressed)
    ui.text_area.disabled = client_connector.generating_lock


def should_we_stop_generating(tokens, hooked_transformer):
    return (not client_connector.should_generate) or (client_connector.stop)


def generate_tokens(text):
    _hook_filter = lambda name: name.endswith("resid_pre")
    with model.hooks(fwd_hooks=[(_hook_filter, add_vector)]):
        new_text = model.generate(
            text,
            max_new_tokens=999999,
            temperature=1,
            verbose=False,
            stop_criterion=should_we_stop_generating,
            new_token_callback=new_token_callback,
        )
    return new_text

    
def main_loop_func():
    while client_connector.receiver_thread.is_alive():
        ui.info_box.value = str(client_connector.pressed) if client_connector.generating_lock else "-"
        if client_connector.should_generate:
            generate_tokens(ui.text_area.value_input)
            # ui.text_area.value_input = generate_tokens(ui.text_area.value_input)
        else:
            time.sleep(0.010)
    ui.info_box.value = "Off"


client_connector.receiver_thread.start()
print("Connector started")

# it needs to be a thread because otherwise panel can't update
main_loop = Thread(target=main_loop_func)
main_loop.start()

print("Serving UI")
ui.full.show(port=args.ui_port)
# ctrl+c will stop this and go past this line

print("\nStopping")
client_connector.stop = True
client_connector.receiver_thread.join()
main_loop.join()
print("Clean end")
