## Installation

### Example conda environment setup
```bash
cd SamGOP
conda create --name samgop python=3.8 -y
conda activate maskdino
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
cd detectron2
pip install -e .
```

### Install Requirements
```bash
cd ..
pip install -r requirements.txt
```


### CUDA kernel for MSDeformAttn
```bash
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
```

