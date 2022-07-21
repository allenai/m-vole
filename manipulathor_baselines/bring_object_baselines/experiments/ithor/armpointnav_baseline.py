import platform

import gym
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import DepthSensorThor, PickedUpObjSensor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import  ExploreWiseRewardTask
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.mvole_sensors import RealPointNavSensor
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.real_pointnav_model import RealPointNavModel


class ArmPointNavBaseline(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    NOISE_LEVEL = 0
    distance_thr = 1.5 
    SENSORS = [
        RGBSensorThor(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorThor(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        PickedUpObjSensor(),
        RealPointNavSensor(type='source'),
        RealPointNavSensor(type='destination'),

    ]

    MAX_STEPS = 200

    TASK_SAMPLER = DiverseBringObjectTaskSampler
    TASK_TYPE = ExploreWiseRewardTask

    NUM_PROCESSES = 40

    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS



    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1  
        self.REWARD_CONFIG['object_found'] = 1  
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['renderInstanceSegmentation'] = False


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return RealPointNavModel(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
            visualize=cls.VISUALIZE
        )

    @classmethod
    def tag(cls):
        return cls.__name__
