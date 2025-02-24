#! /bin/bash
echo "Activating env------------------>"
source /mnt/mydisk/yogesh/anaconda3/etc/profile.d/conda.sh
conda activate env_py_3.10_torch_2.4

echo "Running model: GPT-2"
echo "storing the safe tensor in $HF_HOME"
python gpt2_model.py &> output_gpt2.txt

echo "Deactivating env----------------->"
conda deactivate
