
#!/bin/bash

OUTPUT_FILE="/mnt/drive2/v1dd_stim_analyses"

echo "Stimulus analysis output file: $OUTPUT_FILE"

eval "$(conda shell.bash hook)"
conda activate allen_v1dd

cd .. # cd to allen_v1dd folder

python data_processing/run_stimulus_analyses.py isilon $OUTPUT_FILE --debug
