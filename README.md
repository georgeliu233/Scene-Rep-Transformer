# Scene-Rep-Transformer

This repo is the implementation of the following paper:

**Augmenting Reinforcement Learning with Transformer-based Scene Representation Learning for Decision-making of Autonomous Driving**
<br> [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Zhiyu Huang](https://mczhi.github.io/), [Xiaoyu Mo](https://scholar.google.com/citations?user=JUYVmAQAAAAJ&hl=zh-CN), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[arXiv]](https://arxiv.org/abs/2208.12263)**&nbsp;**[[Project Website]](https://georgeliu233.github.io/Scene-Rep-Transformer/)**

- CARLA Environment is now available;
- Model Framework Overview:
![](pics/main1.png)

## Get started

### 1. Download

* Clone this repository and navigate to the directory:

```
https://github.com/georgeliu233/Scene-Rep-Transformer.git && cd Scene-Rep-Transformer
``` 

* Download required packages:

```
pip install -r requirements.txt
``` 


### 2. Build Scenarios

* Download & build [SMARTS](https://github.com/huawei-noah/SMARTS) according to its repository

* Download official SMARTS scenarios:

```
wget https://github.com/georgeliu233/Scene-Rep-Transformer/releases/download/v1.0.0/smarts_scenarios.tar.gz
```

### Testing Pipelines

### Testing Results:

More testing results in [[Project Website]](https://georgeliu233.github.io/Scene-Rep-Transformer/)

#### Testing results using different rewards

We adopt two extra reward functions for comprehensive testing:


|     Scenario    |     Left turn    |             |                |     Double Merge    |             |                |     CARLA    |             |                |
|-----------------|------------------|-------------|----------------|---------------------|-------------|----------------|--------------|-------------|----------------|
|     Methods     |     Succ.        |     Col.    |     Step(s)    |     Succ.           |     Col.    |     Step(s)    |     Succ.    |     Col.    |     Step(s)    |
|     PPO-R1      |     0.48         |     0.38    |     20.4       |     0.38            |     0.62    |     33.7       |     0.44     |     0.20    |     22.1       |
|     DrQ-R1      |     0.70         |     0.30    |     27.3       |     0.66            |     0.14    |     38.3       |     0.74     |     0.24    |     17.6       |
|     Ours-R1     |     0.90         |     0.10    |     11.7       |     0.84            |     0.16    |     21.2       |     0.76     |     0.22    |     19.4       |
|     PPO-R2      |     0.38         |     0.62    |     31.2       |     0.46            |     0.54    |     34.7       |     0.5      |     0.2o    |     24.8       |
|     DrQ-R2      |     0.82         |     0.08    |     13.9       |     0.72            |     0.28    |     18.7       |     0.78     |     0.12    |     18.5       |
|     Ours-R2     |     0.94         |     0.04    |     11.6       |     0.88            |     0.10    |     27.7       |     0.78     |     0.16    |     21.3       |

## Acknowledgements

RL implementations are based on [tf2rl](https://github.com/keiohta/tf2rl) 

Official release for the strong baselines: [DrQ](https://github.com/denisyarats/drq); [Decision-Transformer](https://github.com/kzl/decision-transformer)
