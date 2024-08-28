# Installation

1. Clone the repo
2. `pip3 install -r requirements.txt` (preferably in a virtual environment)

# Running locally

1. `python3 server.py --model "gpt2-large"` - you can also pass other models listed [here](https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html) - note that some require HuggingFace authentication
2. `python3 key_listener.py`

When the UI opens, you can edit the text in the left pane. Each keyboard letter has a vector assigned to it. When you press `Alt-Shift`, the model starts generating, and for each key you press and hold, this key's vector will be added to the residual stream. You can also tap `Meta/Windows/Command` to toggle the generation so that you don't have to hold `Alt-Shift` all the time.

On startup the vectors of each key are random. But you can assign new, meaningful ones, by writing some text in the right pane, choosing some key, and pressing "Extract". Then, the model will read this text, and the vector will be calculated as the average of residual stream activations over token positions, and then assigned to the key.

# TODO
- [ ] after publishing the post, simplify and document my TL change and make a PR; only one callback func needed, not two

## maybe
- [ ] maybe vast.ai would be cheaper than runpod? look into it
- [ ] use only torch tensors, and no numpy arrays