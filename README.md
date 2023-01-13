# Intrinsic Extrinsic Disentangled Recommendation

## The instructions to create the environment for IEDR.


We run the code on cuda version 11.3 or 11.4, or cpu.

1. Create a conda environment

```
conda create -n py39_torch python=3.9 scipy numpy
conda activate py39_torch
```

2. Install the pytorch 11.2.0

```
# GPU version
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# CPU version
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```


3. Install torch_geometric 

```
# GPU version
CUDA=cu113
TORCH=1.12.0

# CPU version
CUDA=cpu
TORCH=1.12.0

pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

4. Install other packages

```
pip install tensorboard
pip install icecream
pip install numpy
pip install pandas
```

## Run the Code

```
cd run
bash frappe_run.sh
```
