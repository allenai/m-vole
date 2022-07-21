import platform

import gym
import torch
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, \
    CategoryFeatureSampleSensor, DepthSensorThor, PickedUpObjSensor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import ExploreWiseRewardTask
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.mvole_sensors import PointNavEmulatorSensorComplexArm
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_model import RGBDModelWPointNavEmulator


class mVOLEwAgentMotionNoise(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    NOISE_LEVEL = 0
    AGENT_LOCATION_NOISE = 10
    distance_thr = 1.5 
    source_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='source', distance_thr=distance_thr)
    destination_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='destination', distance_thr=distance_thr)
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
        CategorySampleSensor(type='source'),
        CategorySampleSensor(type='destination'),
        CategoryFeatureSampleSensor(type='source'),
        CategoryFeatureSampleSensor(type='destination'),
        source_mask_sensor,
        destination_mask_sensor,
        PointNavEmulatorSensorComplexArm(type='source', mask_sensor=source_mask_sensor, noise=AGENT_LOCATION_NOISE),
        PointNavEmulatorSensorComplexArm(type='destination', mask_sensor=destination_mask_sensor, noise=AGENT_LOCATION_NOISE),

    ]

    MAX_STEPS = 200

    TASK_SAMPLER = DiverseBringObjectTaskSampler
    TASK_TYPE = ExploreWiseRewardTask

    NUM_PROCESSES = 20

    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS


    def train_task_sampler_args(self, **kwargs):
        sampler_args = super(mVOLEwAgentMotionNoise, self).train_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:

            for pointnav_emul_sensor in sampler_args['sensors']:
                if isinstance(pointnav_emul_sensor, PointNavEmulatorSensorComplexArm):
                    pointnav_emul_sensor.device = torch.device(kwargs["devices"][0])

        return sampler_args
    def test_task_sampler_args(self, **kwargs):
        sampler_args = super(mVOLEwAgentMotionNoise, self).test_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:

            for pointnav_emul_sensor in sampler_args['sensors']:
                if isinstance(pointnav_emul_sensor, PointNavEmulatorSensorComplexArm):
                    pointnav_emul_sensor.device = torch.device(kwargs["devices"][0])

        return sampler_args

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1  
        self.REWARD_CONFIG['object_found'] = 1  
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return RGBDModelWPointNavEmulator(
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
