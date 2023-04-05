
#!/bin/bash

OUTPUT_FILE="/home/chase/v1dd_stim_analyses"
LOG_FILE="/home/chase/v1dd_stim_analyses_most_recent_log.txt"

echo "Stimulus analysis output file: $OUTPUT_FILE"

eval "$(conda shell.bash hook)"
conda activate allen_v1dd

python ../data_processing/run_stimulus_analyses.py isilon $OUTPUT_FILE --debug |& tee $LOG_FILE
# python ../data_processing/run_stimulus_analyses.py isilon $OUTPUT_FILE --debug 2>&1 | tee $LOG_FILE