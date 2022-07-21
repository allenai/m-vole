# [Object Manipulation via Visual Target Localization](https://arxiv.org/abs/2203.08141)
#### Kiana Ehsani, Ali Farhadi, Aniruddha Kembhavi, Roozbeh Mottaghi
#### (Presented at ECCV 2022)
#### <a href="https://prior.allenai.org/projects/m-vole">(Project Page)</a>--<a href="https://www.youtube.com/watch?v=otCL-TCzQi4&ab_channel=kianaehsani">(Video)</a>--<a href="">(Slides)</a> 

Object manipulation is a critical skill required for Embodied AI agents interacting with the world around them. Training agents to manipulate objects, poses many challenges. These include occlusion of the target object by the agent's arm, noisy object detection and localization, and the target frequently going out of view as the agent moves around in the scene. We propose Manipulation via Visual Object Location Estimation (m-VOLE), an approach that explores the environment in search for target objects, computes their 3D coordinates once they are located, and then continues to estimate their 3D locations even when the objects are not visible, thus robustly aiding the task of manipulating these objects throughout the episode. Our evaluations show a massive 3x improvement in success rate over a model that has access to the same sensory suite but is trained without the object location estimator, and our analysis shows that our agent is robust to noise in depth perception and agent localization. Importantly, our proposed approach relaxes several assumptions about idealized localization and perception that are commonly employed by recent works in navigation and manipulation -- an important step towards training agents for object manipulation in the real world.

This code base is based on <a href=https://allenact.org/>AllenAct</a> framework and <a href=https://github.com/allenai/manipulathor>ManipulaTHOR</a> framework. The majority of the core training algorithms and pipelines are borrowed from <a href=https://github.com/allenai/manipulathor>ManipulaTHOR code base</a>. 

### Citation

If you find this project useful in your research, please consider citing:

```
   @inproceedings{ehsani2022object,
     title={Object Manipulation via Visual Target Localization},
     author={Ehsani, Kiana and Farhadi, Ali and Kembhavi, Aniruddha and Mottaghi, Roozbeh},
     booktitle={ECCV},
     year={2022}
   }
```

### Contents
<div class="toc">
<ul>
<li><a href="#-installation">💻 Installation</a></li>
<li><a href="#-dataset">📊 Dataset</a></li>
<li><a href="#-training-an-agent">🏋 Training an Agent</a></li>
<li><a href="#-evaluating-a-pre-trained-agent">💪 Evaluating a Pre-Trained Agent</a></li>
</ul>
</div>

## 💻 Installation
 
To begin, clone this repository locally
```bash
git clone https://github.com/allenai/m-vole.git
```
<details>
<summary><b>See here for a summary of the most important files/directories in this repository</b> </summary> 
<p>

Here's a quick summary of the most important files/directories in this repository:
* `manipulathor_utils/*.py` - Helper functions and classes.
* `manipulathor_baselines/bring_object_baselines`
    - `experiments/`
        + `ithor/*.py` - Different baselines introduced in the paper. Each files in this folder corresponds to a row of a table in the paper.
        + `*.py` - The base configuration files which define experiment setup and hyperparameters for training.
    - `models/*.py` - A collection of Actor-Critic baseline models.  
* `ithor_arm/` - A collection of Environments, Task Samplers and Task Definitions
    - `ithor_arm_environment.py` - The definition of the `ManipulaTHOREnvironment` that wraps the AI2THOR-based framework introduced in this work and enables an easy-to-use API.  
    - `itho_arm_constants.py` - Constants used to define the task and parameters of the environment. These include the step size 
      taken by the agent, the unique id of the the THOR build we use, etc.
    - `object_displacement_sensors.py` - Sensors which provide observations to our agents during training.  
    - `object_displacement_tasks.py` - Definition of the `ObjDis` task, the reward definition and the function for calculating the goal achievement. 
    - `object_displacement_task_samplers.py` - Definition of the `ObjDisTaskSampler` samplers. Initializing the sampler, reading the json files from the dataset and randomly choosing a task is defined in this file. 

</p>
</details>

You can then install requirements by running
```bash
pip install -r requirements.txt
```



**Python 3.6+ 🐍.** Each of the actions supports `typing` within <span class="chillMono">Python</span>.

**AI2-THOR <2f8dd9f> 🧞.** To ensure reproducible results, please install this version of the AI2THOR.

After installing the requirements, you should start the xserver by running this command in the background:
```
sudo ai2thor-xorg start
```

## 📊 Dataset

To study the task of ObjDis, we use the ArmPointNav Dataset (APND) presented in <a href=https://github.com/allenai/manipulathor>ManipulaTHOR</a>. This consists of 30 kitchen scenes in AI2-THOR that include more than 150 object categories (69 interactable object categories) with a variety of shapes, sizes and textures. We use 12 pickupable categories as our target objects. We use 20 scenes in the training set and the remaining is evenly split into Val and Test. We train with 6 object categories and use the remaining to test our model in a Novel-Obj setting. To train or evaluate the models download the dataset from [here](https://drive.google.com/file/d/1oPCgOdTD6QbOGHwAGNVr6oP9ToS7zbO7/view?usp=sharing) and extract to `datasets/apnd-dataset`.

###🗂️ Dataset Hierarchy

Below you can find the description and usecase of each of the files in the dataset.

```
apnd-dataset
└── pruned_object_positions ----- Pool of potential object positions
│   └── *_valid_<OBJECT-NAME>_positions_in_<ROOM-ID>.json
└── bring_object_deterministic_tasks ----- Deterministic set of tasks, randomly chosen for evaluation
│   └── tasks_<SOURCE-OBJECT-NAME>_to_<DESTINATION-OBJECT-NAME>_positions_in_<ROOM-ID>.json ----- Consists of pairs of initial and goal location of the object
└── starting_pose.json ----- Initial location in each room for which the arm initialization succeeds
└── valid_agent_initial_locations.json ----- Reachable positions for the agent in each room
└── query_images ----- Canonical Images of Target Objects
└── query_features ----- Imagenet Features of the Canonical Images of Target Objects
```



## 🏋 Training An Agent

For running experiments first you need to add the project directory to your python path. You can train a model with a specific experiment setup by running one of the experiments below:

```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/<EXPERIMENT-NAME>
```

Where `<EXPERIMENT-NAME>` can be one of the options below:

```
armpointnav_baseline -- No Vision Baseline
armpointnav_baseline -- ArmPointNav Baseline
mdm_with_predicted_mask -- MDM Baseline w/ Predicted Mask
loc_mdm_with_predicted_mask -- Loc-MDM Baseline w/ Predicted Mask
mvole_with_predicted_mask -- m-VOLE (Ours) w/ Predicted Mask
``` 


## 💪 Evaluating A Pre-Trained Agent 

To evaluate a pre-trained model, (for example to reproduce the numbers in the paper), you can add
`--eval -c <WEIGHT-ADDRESS>` to the end of the command you ran for training. 

In order to reproduce the numbers in the paper, you need to download the pretrained models from
[here](https://drive.google.com/file/d/1wZi_IL5d7elXLkAb4jOixfY0M6-ZfkGM/view?usp=sharing) and extract them 
to pretrained_models. The full list of experiments and their corresponding trained weights can be found
[here](EvaluateModels.md).

```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/<EXPERIMENT-NAME> -o test_out -s 1 -t test -c <WEIGHT-ADDRESS>
```