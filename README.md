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
We keep an indipendent code strcutures for CARLA and SMARTS, so that you can also choose either of one to install
#### CARLA

* Download all sources of CARLA ```v0.9.13``` via this [link](https://github.com/carla-simulator/carla/releases/tag/0.9.13/)

* Navigate to ```envs/carla/carla_env.py ```, add folder path of the installed CARLA in system in line 18-19:

```
# append sys PATH for CARLA simulator 
# assume xxx is your path to carla
sys.path.append('xxx/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')
sys.path.append('xxx/CARLA_0.9.13/PythonAPI/carla/')
```

#### SMARTS
* Download & build [SMARTS](https://github.com/huawei-noah/SMARTS) according to its repository

* **[NOTE]** The current scenarios are built upon SMARTS ```v0.4.18```, so you may build from [source](https://github.com/huawei-noah/SMARTS/releases/tag/v0.4.17)

* Ensure the SMARTS is correctly build by running ```make sanity-test```

* Download SMARTS Scenarios:

```
wget https://github.com/georgeliu233/Scene-Rep-Transformer/releases/download/v1.0.0/smarts_scenarios.tar.gz
```

### 3. Testing Pipelines

* We offered the [checkpoints](https://github.com/georgeliu233/Scene-Rep-Transformer/releases/download/v1.0.0/data.tar.gz) with ```train_logs``` for all scenarios:

```
wget https://github.com/georgeliu233/Scene-Rep-Transformer/releases/download/v1.0.0/data.tar.gz
```

* unzip the ckpts and scenarios:

```
bash ./tools/download_build.sh
```

* run the scenario test by following example commands:
```
cd tools
python3 test.py \
        --scenario=left_turn # testing scenarios: [left_turn, cross, carla, ..., etc.]
        --algo=scenerep # proposed methods
```

### 4. Testing Results:

More testing results in [[Project Website]](https://georgeliu233.github.io/Scene-Rep-Transformer/)

#### Testing results using different rewards

We adopt two extra reward functions for comprehensive testing:

[[R1]](https://arxiv.org/abs/1904.09503); [[R2]](https://arxiv.org/abs/2005.03863)


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
