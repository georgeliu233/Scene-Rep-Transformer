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
```https://github.com/georgeliu233/Scene-Rep-Transformer.git && cd Scene-Rep-Transformer``` 

* Download dependent packages:

```pip install -r requirements.txt``` 

### Build Scenarios

Download & build [SMARTS](https://github.com/huawei-noah/SMARTS) simulator according to its repository

Build scenarios in ```./smarts/smarts_scenarios```

### Testing Pipelines


## Acknowledgements

RL implementations are based on [tf2rl](https://github.com/keiohta/tf2rl) 

Official release for the strong baselines: [DrQ](https://github.com/denisyarats/drq); [Decision-Transformer](https://github.com/kzl/decision-transformer)
