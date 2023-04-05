#!/bin/bash

SCREEN_NAME="stim_analysis"

echo "Running stimulus analyses on screen \"$SCREEN_NAME\""

cd "$( dirname "$0" )"
screen -dmS $SCREEN_NAME sh stim_analysis_isilon.sh

echo
echo "Type the following to enter screen:"
echo "  screen -r $SCREEN_NAME"