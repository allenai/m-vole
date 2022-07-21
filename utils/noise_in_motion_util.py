import copy
import math

import torch

from utils.noise_from_habitat import ControllerNoiseModel, MotionNoiseModel, _TruncatedMultivariateGaussian
import numpy as np

class NoiseInMotion:
    noise_mode = ControllerNoiseModel(
        linear_motion=MotionNoiseModel(
            _TruncatedMultivariateGaussian([0.074, 0.036], [0.019, 0.033]),
            _TruncatedMultivariateGaussian([0.189], [0.038]),
        ),
        rotational_motion=MotionNoiseModel(
            _TruncatedMultivariateGaussian([0.002, 0.003], [0.0, 0.002]),
            _TruncatedMultivariateGaussian([0.219], [0.019]),
        ),
    )

def tensor_from_dict(pos_dict):
    array = []
    if 'x' in pos_dict:
        array = [pos_dict[k] for k in ['x','y','z']]
    else:
        if 'position' in pos_dict:
            array += [pos_dict['position'][k] for k in ['x','y','z']]
        if 'rotation' in pos_dict:
            array += [pos_dict['rotation'][k] for k in ['x','y','z']]
    return torch.Tensor(array)
def squeeze_bool_mask(mask):
    if type(mask) == np.ndarray:
        mask = mask.astype(bool).squeeze(-1)
    elif type(mask) == torch.Tensor:
        mask = mask.bool().squeeze(-1)
    return mask


def get_accurate_locations(env):
    metadata = copy.deepcopy(env.controller.last_event.metadata)
    camera_xyz = np.array([metadata["cameraPosition"][k] for k in ["x", "y", "z"]])
    camera_rotation=metadata["agent"]["rotation"]["y"]
    camera_horizon=metadata["agent"]["cameraHorizon"]
    arm_state = env.get_absolute_hand_state()

    return dict(camera_xyz=camera_xyz, camera_rotation=camera_rotation, camera_horizon=camera_horizon, arm_state=arm_state)