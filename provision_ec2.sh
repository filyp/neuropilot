#!/bin/bash
git clone https://github.com/filyp/neuropilot.git
cd neuropilot
source activate pytorch
pip install -r requirements.txt
yum -y install htop
# this rc.local thing is not working
echo "export BOKEH_ALLOW_WS_ORIGIN='*.eu-north-1.compute.amazonaws.com:443'" >> /etc/rc.local