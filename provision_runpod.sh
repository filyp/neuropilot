#!/bin/bash
cd /workspace    # so that it can get persisted when I stop the pod
git clone https://github.com/filyp/neuropilot.git
cd neuropilot
pip install -r requirements.txt

apt update
apt -y install htop

# running
export BOKEH_ALLOW_WS_ORIGIN='*'
wget https://raw.githubusercontent.com/facebookresearch/llama/main/download.sh
bash download.sh