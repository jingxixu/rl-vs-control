# How Are Learned Perception-Based Controllers Impacted by the Limits of Robust Control?

[Jingxi Xu](https://jxu.ai),
[Bruce Lee](https://brucedlee.github.io/),
[Dinesh Jayaraman](https://www.seas.upenn.edu/~dineshj/),
[Nikolai Matni](https://nikolaimatni.github.io/),
<br>
Columbia University, New York, NY, United States<br>
University of Pennsylvania, Philadelphia, PA, United States<br>
[L4DC 2021](https://l4dc.ethz.ch/)

### [Project Page](https://jxu.ai/rl-vs-control-web/) | [Arxiv](https://arxiv.org/abs/2104.00827)

  <p align="left">
  <img src="imgs/cartpole.png" width=600>
  </p>

## Setup

We use Anaconda for building the virtual environment for this project

```
conda create --name rlvscontrol --file spec-file.txt
```

You can activate and deactivate your envrionment with

```
conda activate rlvscontrol
conda deactivate
```

After the envrionment is activated, install some extra packages with

```
pip install gym pybullet transformations opencv-python
```

## Perception Model

To collect data for training the perception model, run

```
python collect_perception_data.py --fixation [FIXATION] --num_samples [NUM_SAMPLES] --save_dir [SAVE_DIR]
```

To train the perception model with depth images, run

```
python train_perception_model --dataset_path [DATASET_PATH] --fixation [FIXATION]
```

To train the perception model with RGB images, run

```
python train_perception_model --dataset_path [DATASET_PATH] --fixation [FIXATION] --rgb
```

We have provided you with the already trained perception model checkpoints for you to use directly in `assets/models/perception/z/`.

## Reinforcement Learning

This section provides commands to train the soft-actor-critic RL agent

To train the agent with noise-free z perception model, run

```
python train_rl_sac.py --ob_type z_sequence --fixation [FIXATION]
```

To train the agent with depth image perception model, run

```
python train_rl_sac.py --ob_type z_sequence --fixation [FIXATION] --pnoise --perception_model_path [PERCEPTION_MODEL_PATH]
```

To train the agent with RGB image perception model, run

```
python train_rl_sac.py --ob_type z_sequence --fixation [FIXATION] --pnoise --perception_model_path [PERCEPTION_MODEL_PATH] --rgb
```

## H Infinity Control

TBD

## Citation

```
@inproceedings{xu2021rlvscontrol,
	title={How Are Learned Perception-Based Controllers Impacted by the Limits of Robust Control?},
	author={Xu, Jingxi and Lee, Bruce and Matni, Nikolai and Jayaraman, Dinesh},
	booktitle={Learning for Dynamics and Control (L4DC)},
	year={2021}
}
```
