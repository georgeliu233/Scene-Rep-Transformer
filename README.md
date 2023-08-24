# Augmenting Reinforcement Learning with Transformer-based Scene Representation Learning for Decision-making of Autonomous Driving
[Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Zhiyu Huang](https://mczhi.github.io/), [Xiaoyu Mo](https://scholar.google.com/citations?user=JUYVmAQAAAAJ&hl=zh-CN), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 

[AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)

## Abstract 
Decision-making for urban autonomous driving is challenging due to the stochastic nature of interactive traffic participants and the complexity of road structures. Although reinforcement learning (RL)-based decision-making schemes are promising to handle urban driving scenarios, they suffer from low sample efficiency and poor adaptability. In this paper, we propose Scene-Rep Transformer to enhance the RL decision-making capabilities through improved scene representation encoding and sequential predictive latent distillation. Specifically, a multi-stage Transformer (MST) encoder is constructed to model not only the interaction awareness between the ego vehicle and its neighbors but also intention awareness between the agents and their candidate routes. A sequential latent Transformer (SLT) with self-supervised learning objectives is employed to distill future predictive information into the latent scene representation, in order to reduce the exploration space and speed up training. The final decision-making module based on soft actor-critic (SAC) takes as input the refined latent scene representation from the Scene-Rep Transformer and generates decisions. The framework is validated in five challenging simulated urban scenarios with dense traffic, and its performance is manifested quantitatively by substantial improvements in data efficiency and performance in terms of success rate, safety, and efficiency. Qualitative results reveal that our framework is able to extract the intentions of neighbor agents, enabling better decision-making and more diversified driving behaviors.

## Method Overview 
<img src="./pics/main1.png" style="width:90%;">

An overview of our RL decision-making framework with Scene-Rep Transformer. Given perception-processed vectorized scene inputs, the multi-stage Transformer (MST) encodes the multi-modal information with interaction awareness. During training, a sequential latent Transformer (SLT) performs representation learning using consecutive latent-action pairs to ensure future consistency. The soft-actor-critic (SAC) module takes the latent feature vector to make driving decisions for downstream planning and control tasks. 

## Results

Here we present the testing results from different methods (2x speed). 

### CARLA Urban unsignalized left turn

| Scene-Rep Transformer | Drq | PPO |
|:-----------------:|:-----------------:|:-----------------:|
| <img src="./pics/sel0_out.gif" style="width:90%;"> | <img src="./pics/aggressive+(1)_out.gif" style="width:90%;"> | <img src="./pics/ppo_out.gif" style="width:90%;"> |


### SMARTS Unprotected left turn 

### SMARTS Double merging

### Testing results using different rewards
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
