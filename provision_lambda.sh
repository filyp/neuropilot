#!/bin/bash
git clone https://github.com/filyp/neuropilot.git
cd neuropilot
# without venv there are conflicts with existing packages
python -m venv venv
pip install -r requirements.txt

# running
sudo su   # required to bind to port 443; hmmm what about using some different port later?
export BOKEH_ALLOW_WS_ORIGIN='*:443'