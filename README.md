# NeuLNS
Neural Large Neighborhood Search

Learn to Design Heuristics for Vehicle Routing Problem (VRP), by Deep Learning and Reinforcement Learning. This project provides the code to replicate the experiments in the paper:

> <cite> Learn to Design the Heuristics for Vehicle Routing Problem [arxiv link](https://arxiv.org/abs/2002.08539)</cite>

Welcome to cite our work (bib):

``` 
@misc{gao2020learn,
    title={Learn to Design the Heuristics for Vehicle Routing Problem},
    author={Lei Gao and Mingxiang Chen and Qichang Chen and Ganzhong Luo and Nuoyi Zhu and Zhixin Liu},
    year={2020},
    eprint={2002.08539},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```

Please install vrp_env-0.1.1 before training or evaluation. run train_model.py
to train a cvrp/cvrptw model, and evaluation.py to evaluate on the test data. The
default arguments can be found in arguments.py.

Example:
```
python train_model.py -n 99 -c 100
```
By 物界科技 WaterMirror Ltd. www.water-mirror.com

**We Are Hiring!! ML, OR, please email at liuzhixin\@watermirror.ai**
