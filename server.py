# original activation engineering code: https://colab.research.google.com/drive/1y84fhgkGX0ft2DmYJB3K13lAyf-0YonK?usp=sharing#scrollTo=ZExJFurIjKHM
# some supported models: https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=jHj79Pj58cgJKdq4t-ygK-4h
import time
import socket
import json
import argparse
from threading import Thread
from typing import Any, Dict, Union, List, Tuple

import hashlib
import matplotlib
import matplotlib.pyplot as plt
import panel as pn
import torch
import numpy as np
from transformer_lens import HookedTransformer


parser = argparse.ArgumentParser(description="Run the server for activation engineering")
parser.add_argument("--ui_port", type=int, default=5000, help="Port for the UI")
parser.add_argument("--connector_port", type=int, default=5005, help="Port for the connector")
parser.add_argument("--model", type=str, default="gpt2-small", help="Name of the model to use")
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
    def __init__(self, num_layers: int, directions_data: Dict[str, Dict[str, Any]]):
        self._directions_data = directions_data
        self.text_area = pn.widgets.TextAreaInput(value="", sizing_mode="stretch_both")
        self.steering_strength = pn.widgets.FloatSlider(
            name="Steering Strength (log10)", start=-3, end=3, step=0.01, value=0
        )
        self.layer_num = pn.widgets.IntSlider(name="Layer Number", start=0, end=num_layers - 1, step=1, value=6)
        self.info_box = pn.widgets.StaticText(name="Info", value="Not started")
        self.letter_select = pn.widgets.Select(name="Letter", options=list(directions_data.keys()))
        self.letter_select.param.watch(self._update_text_to_extract_area, "value")
        self.extract_direction_button = pn.widgets.Button(name="Extract", button_type="primary", align="end")
        self.extract_direction_button.on_click(self._extract_direction)
        self.text_to_extract_area = pn.widgets.TextAreaInput(value="", sizing_mode="stretch_both")
        self._update_text_to_extract_area(None)

        # some square for 2d plotting, must have square aspect ratio
        # turn interactive plotting off, so that it's not displayed in notebook
        plt.ioff()
        fig, ax = plt.subplots(figsize=(20, 20))

        ax.set_aspect("equal")  # note: not sure if clear resets this or not
        ax.set_facecolor((0.0, 0.0, 0.0))  # black background
        self._ax = ax
        self._max_plotting_scale = 0.000001
        self.update_plot(None, None)  # set up plot
        # self.plot = pn.pane.Matplotlib(fig, tight=True, format="svg", height=400, width=400)
        self.plot = pn.pane.Matplotlib(fig, tight=True, format="svg", sizing_mode="stretch_height")
        # svg format is necessary; without it there are some weird lags when updating the text!

        self.full = pn.Row(
            self.text_area,
            pn.Column(
                pn.Row(self.letter_select, self.extract_direction_button, sizing_mode="stretch_width"),
                self.text_to_extract_area,
                pn.Row(
                    pn.Column(
                        self.info_box,
                        self.steering_strength,
                        self.layer_num,
                    ),
                    # add a spacer to push the plot to the right maximally
                    pn.Spacer(sizing_mode="stretch_width"),
                    self.plot,
                    sizing_mode="stretch_width",
                ),
                # pn.FlexBox(
                #     pn.Column(
                #         self.info_box,
                #         self.steering_strength,
                #         self.layer_num,
                #     ),
                #     self.plot,
                #     flex_direction="row",
                #     justify_content="space-between",
                # ),
            ),
        )

    def update_plot(
        self,
        existing_activation: List[float],
        modifying_activations: List[Tuple[List[float], str]],
    ):
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
            hue = int(hashlib.shake_128(key.encode("utf-8")).hexdigest(1), 16)
            color = matplotlib.colors.hsv_to_rgb((hue / 255, 1, 1))

            # ax.arrow(vector_sum[0] / s, vector_sum[1] / s, activation[0] / s, activation[1] / s, color="red", linewidth=2, head_width=0.02, head_length=0.02)
            new_vector_sum = vector_sum + np.array(activation[:2])
            ax.plot(
                [vector_sum[0] / s, new_vector_sum[0] / s],
                [vector_sum[1] / s, new_vector_sum[1] / s],
                color=color,
                linewidth=2,
            )
            vector_sum = new_vector_sum

        # update plot
        self.plot.param.trigger("object")

    def _update_text_to_extract_area(self, event):
        letter = self.letter_select.value
        self.text_to_extract_area.value = self._directions_data[letter]["source_text"]

    def _extract_direction(self, event):
        generator.extract_direction(
            self.text_to_extract_area.value,
            self.letter_select.value,
        )


class Generator:
    def __init__(self, new_token_callback):
        self.new_token_callback = new_token_callback

        # load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)  # save memory

        if "llama" in args.model.lower():
            raise NotImplementedError("Llama models are not supported yet")
        else:
            self.model = HookedTransformer.from_pretrained(args.model, device=self.device)

        self.num_layers = len(self.model._modules["blocks"])

        # consctruct a random set of directions
        np.random.seed(0)
        self.directions_data = dict()
        for letter in "abcdefghijklmnopqrstuvwxyz,.":
            self.directions_data[letter] = dict(
                direction=np.random.normal(0, 1, self.model.cfg.d_model),
                source_text="<random>",
            )
        # TODO handling these directions could be done by a class, together with looping over pressed keys, and later extracting directions from text

    def add_vector(self, resid_pre, hook):
        # this function will be run as a hook
        to_add = np.zeros(self.model.cfg.d_model)
        modifying_activations = []
        for key in client_connector.pressed:
            if key in self.directions_data:
                component = self.directions_data[key]["direction"] * (10**ui.steering_strength.value)
                to_add += component
                modifying_activations.append((component[:2], key))

        ui.update_plot(resid_pre[:, -1, :2].flatten().cpu(), modifying_activations)

        resid_pre[:, -1, :] += torch.from_numpy(to_add).to(self.device)
        # TODO double check that this broadcasting works as intended

    def generate_tokens(self, text):
        _hook_filter = lambda name: name == f"blocks.{ui.layer_num.value}.hook_resid_pre"
        with self.model.hooks(fwd_hooks=[(_hook_filter, self.add_vector)]):
            new_text = self.model.generate(
                text,
                max_new_tokens=999999,
                temperature=1,
                verbose=False,
                stop_criterion=self.new_token_callback,
            )
        return new_text

    def extract_direction(self, text, letter):
        ui.info_box.value = "Extracting..."
        name = f"blocks.{ui.layer_num.value}.hook_resid_pre"
        cache, caching_hooks, _ = self.model.get_caching_hooks(lambda n: n == name)
        with self.model.hooks(fwd_hooks=caching_hooks):
            _ = self.model(text)
        activations = cache[name][0]

        self.directions_data[letter] = dict(
            direction=activations.mean(dim=0).cpu().numpy(),
            source_text=text,
            hook_point=name,
            activations=activations.cpu().numpy(),
        )
        ui.info_box.value = "Extracted"


def new_token_callback(tokens, hooked_transformer):
    tokens_to_display = tokens[0][1:]  # remove the BOS token TODO: but this is not always the case? we should check
    text = hooked_transformer.tokenizer.decode(tokens_to_display)
    ui.text_area.value_input = text
    # btw, update all the rest
    ui.info_box.value = str(client_connector.pressed)
    ui.text_area.disabled = client_connector.generating_lock

    # return True if we should stop generating
    return (not client_connector.should_generate) or (client_connector.stop)


generator = Generator(new_token_callback)
client_connector = ClientConnector()
ui = UI(num_layers=generator.num_layers, directions_data=generator.directions_data)


def main_loop_func():
    while client_connector.receiver_thread.is_alive():
        ui.info_box.value = str(client_connector.pressed) if client_connector.generating_lock else "-"
        if client_connector.should_generate:
            generator.generate_tokens(ui.text_area.value_input)
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
# websocket_origin=["*.ngrok.io", "localhost:5000"],
# ctrl+c will stop this and go past this line

print("\nStopping")
client_connector.stop = True
client_connector.receiver_thread.join()
main_loop.join()
print("Clean end")
