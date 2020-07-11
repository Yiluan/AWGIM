## Attentive Weights Generation for Few Shot Learning via Information Maximization

Published at CVPR 2020

By Yiluan Guo, Ngai-Man Cheung

[Paper Link](http://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Attentive_Weights_Generation_for_Few_Shot_Learning_via_Information_Maximization_CVPR_2020_paper.pdf)

The implementation is written in Python 3 and has been tested on tensorflow 1.12.0, Ubuntu 16.04. 

Parts of the code are borrowed from [LEO](https://github.com/deepmind/leo). 

The feature embeddings for miniImageNet and tieredImageNet can be downloaded from https://github.com/deepmind/leo.

5-way 1-shot experiment on miniImageNet:
 
`python main.py`


The hyper-parameters can be tuned in `main.py` and AWGIM is in `model.py`.


### Citation
Please cite our work if you find it useful in your research:
```
@inproceedings{guo2020awgim,
  title = {Attentive Weights Generation for Few Shot Learning via Information Maximization},
  author = {Yiluan Guo, Ngai-Man Cheung},
  booktitle = {CVPR},
  year = {2020}
}
```
