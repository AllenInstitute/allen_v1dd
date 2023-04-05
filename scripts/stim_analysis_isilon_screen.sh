#!/bin/bash

SCREEN_NAME="stim_analysis"
LOG_FILE="/home/chase/v1dd_stim_analyses_most_recent_log.txt"

echo "Running stimulus analyses on screen \"$SCREEN_NAME\""

cd "$( dirname "$0" )"

# screen -dmS $SCREEN_NAME bash -c "./stim_analysis_isilon.sh |& tee $LOG_FILE" # ; exec bash
screen -dmS $SCREEN_NAME bash -c "./stim_analysis_isilon.sh; exec bash"

echo
echo "  ENTER SCREEN:  screen -r $SCREEN_NAME"
# echo "  less $LOG_FILE                  to view log file"
echo
