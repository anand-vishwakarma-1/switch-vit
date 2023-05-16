# HPML Project - Model scaling with switch transformer methodology for transformer-based image classification models

Team:<br>
| | |
|---|---|
| Raj Ghodasara | rg4357 |
| Anand Vishwakarma | asv8775 |

Project Description and observations - [Final Presentation.pdf](Final%20Presentation.pdf)

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


`python vit.py --batch 256 --experts 64 --epochs 500 --gpu 4 --dmodel 300 --noswitch --out vit_noswitch_model_rtx_4_dmodel_300_batch_256_cifar10`<br>
Running original vit with 64 experts and 300 d_model embdedding size on 4 GPUs with 256 effective batch size over multiple GPU.


# Results
model trained on cifar 10 for 350 epochs<br>
GPU: 4 RTX8000<br>
Batch size: 256<br>
Embedding Size: 300<br>
Patch Size: 4

t_epoch = Epoch to accuracy threshold 75%
| model | experts | size | highest acc (epoch) | t_epoch | Training Time (60,000 images) | Inference Time (60,000 images) |
|----|:----:|:----:|:-----:|:-----:|:-----:|:-----:|
| vit | - | 12,798,490 | 81.36 (345) | 31  | 11.8 | 4.6 |
| switch-vit | 32 | 238,732,426 | 79.04 (321) | 33 | 68.86 | 4.98 |
| switch-vit | 64 | 464,669,962 | 78.8 (328) | 49 | 130.52 | 5.18 |


# Citation
```
{
 author = {William Fedus, Barret Zoph, Noam Shazeer},
 title = {Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity},
 year = {2022},
 url = {https://arxiv.org/pdf/2101.03961.pdf},
}

@misc{labml,
 author = {Varuna Jayasiri, Nipun Wijerathne},
 title = {labml.ai: A library to organize machine learning experiments},
 year = {2020},
 url = {https://labml.ai/},
}
```
