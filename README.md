# HPML Project - Model scaling with switch transformer methodology for transformer-based image classification models

Team:<br>
Raj Ghodasara&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;- rg4357<br>
Anand Vishwakarma &nbsp;- asv8775<br>

# Usage Example
## usage help
```
usage: vit.py [-h] [--gpu GPU] [--epochs EPOCHS] [--experts EXPERTS] [--batch BATCH] [--noswitch] [--cifar100] [--out OUT] [--dmodel DMODEL]

PyTorch Distributed deep learning

options:
  -h, --help         show this help message and exit
  --gpu GPU          no. of gpus
  --epochs EPOCHS    no. of epochs
  --experts EXPERTS  no. of experts
  --batch BATCH      batch size
  --noswitch         use original vit
  --cifar100         use cifar 100 dataset
  --out OUT          model output path
  --dmodel DMODEL    d_model embedding size
```

## examples

`python vit.py --batch 256 --experts 64 --epochs 500 --gpu 4 --dmodel 300 --out vit_model_rtx_4_dmodel_300_experts_64_batch_256_cifar10`<br>
Running switch-vit with 64 experts and 300 d_model embdedding size on 4 GPUs with 256 effective batch size over multiple GPU.


`python vit.py --batch 256 --experts 64 --epochs 500 --gpu 4 --dmodel 300 --noswitch --out vit_noswitch_model_rtx_4_dmodel_300_experts_64_batch_256_cifar10`<br>
Running original vit with 64 experts and 300 d_model embdedding size on 4 GPUs with 256 effective batch size over multiple GPU.

