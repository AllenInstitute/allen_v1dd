#!/bin/bash

SCREEN_NAME="stim_analysis"
OUTPUT_FILE="~/v1dd_stim_analyses"

echo "Stimulus analysis output file: $OUTPUT_FILE"
echo "Running stimulus analyses on screen \"$SCREEN_NAME\"..."
screen -dmS $SCREEN_NAME python data_processing/run_stimulus_analyses.py --test_mode isilon $OUTPUT_FILE --debug
echo
echo "Type the following to enter screen:"
echo "  screen -r $SCREEN_NAME"