#!/bin/bash

SCREEN_NAME="stim_analysis"
OUTPUT_FILE="/homes/chase/v1dd_stim_analyses"
LOG_FILE="/Homes/chase/v1dd_stim_analyses_most_recent_log.txt"

echo "Stimulus analysis output file: $OUTPUT_FILE"
echo "Running stimulus analyses on screen \"$SCREEN_NAME\""
screen -dmS $SCREEN_NAME `python data_processing/run_stimulus_analyses.py isilon $OUTPUT_FILE --debug |& tee $LOG_FILE`
echo
echo "Type the following to enter screen:"
echo "  screen -r $SCREEN_NAME"