from typing import Sequence, Union

import gym
import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder

from manipulathor_baselines.bring_object_baselines.experiments.bring_object_base import BringObjectBaseConfig
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_model import RGBDModelWPointNavEmulator


class BringObjectMixInSimpleGRUConfig(BringObjectBaseConfig):
    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        return preprocessors

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
