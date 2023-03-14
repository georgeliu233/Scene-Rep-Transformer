# Scene-Rep-Transformer

This repo is the implementation of the following paper:

**Augmenting Reinforcement Learning with Transformer-based Scene Representation Learning for Decision-making of Autonomous Driving**
<br> [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Zhiyu Huang](https://mczhi.github.io/), [Xiaoyu Mo](https://scholar.google.com/citations?user=JUYVmAQAAAAJ&hl=zh-CN), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[arXiv]](https://arxiv.org/abs/2208.12263)**&nbsp;

- CARLA Test will be released soon (undergoing internal scrutiny);
- Model Framework Overview:
![](pics/main1.png)

## Get started

Download dependencies from ```tf2rl``` 

### Simulations

Download & build [SMARTS](https://github.com/huawei-noah/SMARTS) simulator according to its repository

Build scenarios in ```./smarts/smarts_scenarios```

### Experiment Pipelines

1. Rule-based driver model: ```./smarts/test_rule_based.py```

2. On-policy baselines:  ```./smarts/ppo_baseline.py```

3. Our methods:  ```./smarts/train_sac_map.py```

4. Decision-Transformer:  

- Offline data collections: ```./smarts/DT/collect_data.py ```

- Training & Eval: ```./smarts/DT/gym/experiment.py```

5. DrQ: ```./smarts/Drq/train.py```

## Acknowledgements

RL implementations are based on [tf2rl](https://github.com/keiohta/tf2rl) 

Official release for the strong baselines: [DrQ](https://github.com/denisyarats/drq); [Decision-Transformer](https://github.com/kzl/decision-transformer)
