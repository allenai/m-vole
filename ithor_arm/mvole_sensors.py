"""Utility classes and functions for sensory inputs used by the models."""
import copy
import math
import random
from typing import Any, Optional

import cv2
import gym
import numpy as np
import torch

from allenact.embodiedai.mapping.mapping_utils.point_cloud_utils import depth_frame_to_world_space_xyz, project_point_cloud_to_map
from allenact.embodiedai.sensors.vision_sensors import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment


from torch.distributions.utils import lazy_property

from ithor_arm.arm_calculation_utils import convert_world_to_agent_coordinate, diff_position, convert_state_to_tensor
from ithor_arm.bring_object_sensors import add_mask_noise, DepthSensorThor
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from manipulathor_baselines.bring_object_baselines.models.detection_model import ConditionalDetectionModel

from utils.noise_depth_util_files.sim_depth import RedwoodDepthNoise
from utils.noise_from_habitat import ControllerNoiseModel, MotionNoiseModel, _TruncatedMultivariateGaussian
from utils.noise_in_motion_util import NoiseInMotion, squeeze_bool_mask, tensor_from_dict


class PointNavEmulatorSensor(Sensor):

    def __init__(self, type: str, mask_sensor:Sensor,  uuid: str = "point_nav_emul", noise = 0, **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.type = type
        self.mask_sensor = mask_sensor
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.dummy_answer = torch.zeros(3)
        self.dummy_answer[:] = 4 # is this good enough?
        self.device = torch.device("cpu")
        self.min_xyz = np.zeros((3))
        self.noise_mode = ControllerNoiseModel(
            linear_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.074, 0.036], [0.019, 0.033]),
                _TruncatedMultivariateGaussian([0.189], [0.038]),
            ),
            rotational_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.002, 0.003], [0.0, 0.002]),
                _TruncatedMultivariateGaussian([0.219], [0.019]),
            ),
        )


        super().__init__(**prepare_locals_for_super(locals()))


    def get_accurate_locations(self, env):
        metadata = copy.deepcopy(env.controller.last_event.metadata)
        camera_xyz = np.array([metadata["cameraPosition"][k] for k in ["x", "y", "z"]])
        camera_rotation=metadata["agent"]["rotation"]["y"]
        camera_horizon=metadata["agent"]["cameraHorizon"]
        arm_state = env.get_absolute_hand_state()
        return dict(camera_xyz=camera_xyz, camera_rotation=camera_rotation, camera_horizon=camera_horizon, arm_state=arm_state)


    def add_translation_noise(self, change_in_xyz, prev_location):

        if np.abs(change_in_xyz).sum() > 0:

            noise_value_x, noise_value_z = self.noise_mode.linear_motion.linear.sample() * 0.01 * self.noise #to convert to meters
            new_change_in_xyz = change_in_xyz.copy()
            new_change_in_xyz[0] += noise_value_x
            new_change_in_xyz[2] += noise_value_z
            real_rotation = self.real_prev_location['camera_rotation']
            belief_rotation = self.belief_prev_location['camera_rotation']
            diff_in_rotation = math.radians(belief_rotation - real_rotation)
            # 𝑥2=cos𝛽𝑥1−sin𝛽𝑦1
            # 𝑦2=sin𝛽𝑥1+cos𝛽𝑦1
            new_location = prev_location.copy()
            x = math.cos(diff_in_rotation) * new_change_in_xyz[0] - math.sin(diff_in_rotation) * new_change_in_xyz[2]
            z = math.sin(diff_in_rotation) * new_change_in_xyz[0] + math.cos(diff_in_rotation) * new_change_in_xyz[2]
            new_location[0] += x
            new_location[2] += z
        else:
            new_location = prev_location + change_in_xyz
        return new_location
    def rotate_x_z_around_center(self, x, z, rotation):

        new_x = math.cos(rotation) * x - math.sin(rotation) * z
        new_z = math.sin(rotation) * x + math.cos(rotation) * z

        return new_x, new_z
    def add_rotation_noise(self, change_in_rotation, prev_rotation):
        new_rotation = prev_rotation + change_in_rotation

        if change_in_rotation > 0:
            noise_in_rotation = self.noise_mode.rotational_motion.rotation.sample().item()
            new_rotation += noise_in_rotation
        return new_rotation


    def get_agent_localizations(self, env):

        if self.noise == 0:
            return self.get_accurate_locations(env)
        else:
            real_current_location = self.get_accurate_locations(env)

            if self.real_prev_location is None:
                self.real_prev_location = copy.deepcopy(real_current_location)
                self.belief_prev_location = copy.deepcopy(real_current_location)
            else:

                belief_camera_horizon = real_current_location['camera_horizon']
                change_in_xyz = real_current_location['camera_xyz'] - self.real_prev_location['camera_xyz']
                change_in_rotation = real_current_location['camera_rotation'] - self.real_prev_location['camera_rotation']
                belief_camera_xyz = self.add_translation_noise(change_in_xyz, self.belief_prev_location['camera_xyz'])
                belief_camera_rotation = self.add_rotation_noise(change_in_rotation, self.belief_prev_location['camera_rotation'])
                belief_arm_state = real_current_location['arm_state']

                self.belief_prev_location = copy.deepcopy(dict(camera_xyz=belief_camera_xyz, camera_rotation=belief_camera_rotation, camera_horizon=belief_camera_horizon, arm_state=belief_arm_state))
                self.real_prev_location = copy.deepcopy(real_current_location)


            return self.belief_prev_location
    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        mask = squeeze_bool_mask(self.mask_sensor.get_observation(env, task, *args, **kwargs))
        depth_frame = env.controller.last_event.depth_frame.copy()
        depth_frame[~mask] = -1

        if task.num_steps_taken() == 0:
            self.pointnav_history_aggr = []
            self.real_prev_location = None
            self.belief_prev_location = None


        agent_locations = self.get_agent_localizations(env)

        camera_xyz = agent_locations['camera_xyz']
        camera_rotation = agent_locations['camera_rotation']
        camera_horizon = agent_locations['camera_horizon']
        arm_state = agent_locations['arm_state']

        fov = env.controller.last_event.metadata['fov']


        if mask.sum() != 0:
            world_space_point_cloud = calc_world_coordinates(self.min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, self.device, depth_frame)
            valid_points = (world_space_point_cloud == world_space_point_cloud).sum(dim=-1) == 3
            point_in_world = world_space_point_cloud[valid_points]
            middle_of_object = point_in_world.mean(dim=0)
            self.pointnav_history_aggr.append((middle_of_object.cpu(), len(point_in_world)))

        return self.average_so_far(camera_xyz, camera_rotation, arm_state)


    def average_so_far(self, camera_xyz, camera_rotation, arm_state):
        if len(self.pointnav_history_aggr) == 0:
            return self.dummy_answer
        else:
            if self.noise == 0:
                total_sum = [k * v for k,v in self.pointnav_history_aggr]
                total_sum = sum(total_sum)
                total_count = sum([v for k,v in self.pointnav_history_aggr])
                midpoint = total_sum / total_count
                self.pointnav_history_aggr = [(midpoint.cpu(), total_count)]

            else:

                timed_weights = [i + 1 for i in range(len(self.pointnav_history_aggr))]
                total_sum = [timed_weights[i] * self.pointnav_history_aggr[i][0] * self.pointnav_history_aggr[i][1] for i in range(len(self.pointnav_history_aggr))]
                total_count = [v for k,v in self.pointnav_history_aggr]
                real_total_count = [total_count[i] * timed_weights[i] for i in range(len(total_count))]
                midpoint = sum(total_sum) / sum(real_total_count)
                midpoint = midpoint.cpu()

            agent_state = dict(position=dict(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2], ), rotation=dict(x=0, y=camera_rotation, z=0))
            midpoint_position_rotation = dict(position=dict(x=midpoint[0], y=midpoint[1], z=midpoint[2]), rotation=dict(x=0,y=0,z=0))
            midpoint_agent_coord = convert_world_to_agent_coordinate(midpoint_position_rotation, agent_state)

            arm_state_agent_coord = convert_world_to_agent_coordinate(arm_state, agent_state)
            distance_in_agent_coord = dict(x=arm_state_agent_coord['position']['x'] - midpoint_agent_coord['position']['x'],y=arm_state_agent_coord['position']['y'] - midpoint_agent_coord['position']['y'],z=arm_state_agent_coord['position']['z'] - midpoint_agent_coord['position']['z'])

            agent_centric_middle_of_object = torch.Tensor([distance_in_agent_coord['x'], distance_in_agent_coord['y'], distance_in_agent_coord['z']])

            # Removing this hurts the performance
            agent_centric_middle_of_object = agent_centric_middle_of_object.abs()
            return agent_centric_middle_of_object

class PointNavEmulatorSensorComplexArm(PointNavEmulatorSensor):

    def average_so_far(self, camera_xyz, camera_rotation, arm_state):
        if len(self.pointnav_history_aggr) == 0:
            return self.dummy_answer
        else:
            if self.noise == 0:
                total_sum = [k * v for k,v in self.pointnav_history_aggr]
                total_sum = sum(total_sum)
                total_count = sum([v for k,v in self.pointnav_history_aggr])
                midpoint = total_sum / total_count
                self.pointnav_history_aggr = [(midpoint.cpu(), total_count)]

            else:

                timed_weights = [i + 1 for i in range(len(self.pointnav_history_aggr))]
                total_sum = [timed_weights[i] * self.pointnav_history_aggr[i][0] * self.pointnav_history_aggr[i][1] for i in range(len(self.pointnav_history_aggr))]
                total_count = [v for k,v in self.pointnav_history_aggr]
                real_total_count = [total_count[i] * timed_weights[i] for i in range(len(total_count))]
                midpoint = sum(total_sum) / sum(real_total_count)
                midpoint = midpoint.cpu()

            real_arm_state = self.real_prev_location['arm_state']
            real_camera_xyz = self.real_prev_location['camera_xyz']
            real_camera_rotation = self.real_prev_location['camera_rotation']
            real_agent_state = dict(position=dict(x=real_camera_xyz[0], y=real_camera_xyz[1], z=real_camera_xyz[2], ), rotation=dict(x=0, y=real_camera_rotation, z=0))
            real_arm_state_agent_coord = convert_world_to_agent_coordinate(real_arm_state, real_agent_state)

            agent_state = dict(position=dict(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2], ), rotation=dict(x=0, y=camera_rotation, z=0))
            midpoint_position_rotation = dict(position=dict(x=midpoint[0], y=midpoint[1], z=midpoint[2]), rotation=dict(x=0,y=0,z=0))
            midpoint_agent_coord = convert_world_to_agent_coordinate(midpoint_position_rotation, agent_state)

            distance_in_agent_coord = dict(x=real_arm_state_agent_coord['position']['x'] - midpoint_agent_coord['position']['x'],y=real_arm_state_agent_coord['position']['y'] - midpoint_agent_coord['position']['y'],z=real_arm_state_agent_coord['position']['z'] - midpoint_agent_coord['position']['z'])

            agent_centric_middle_of_object = torch.Tensor([distance_in_agent_coord['x'], distance_in_agent_coord['y'], distance_in_agent_coord['z']])

            agent_centric_middle_of_object = agent_centric_middle_of_object.abs()
            return agent_centric_middle_of_object
class PointNavEmulatorSensorOnlyAgentLocation(PointNavEmulatorSensor):

    def average_so_far(self, camera_xyz, camera_rotation, arm_state):
        if len(self.pointnav_history_aggr) == 0:
            return self.dummy_answer
        else:
            if self.noise == 0:
                total_sum = [k * v for k,v in self.pointnav_history_aggr]
                total_sum = sum(total_sum)
                total_count = sum([v for k,v in self.pointnav_history_aggr])
                midpoint = total_sum / total_count
                self.pointnav_history_aggr = [(midpoint.cpu(), total_count)]

            else:

                timed_weights = [i + 1 for i in range(len(self.pointnav_history_aggr))]
                total_sum = [timed_weights[i] * self.pointnav_history_aggr[i][0] * self.pointnav_history_aggr[i][1] for i in range(len(self.pointnav_history_aggr))]
                total_count = [v for k,v in self.pointnav_history_aggr]
                real_total_count = [total_count[i] * timed_weights[i] for i in range(len(total_count))]
                midpoint = sum(total_sum) / sum(real_total_count)
                midpoint = midpoint.cpu()

            agent_state = dict(position=dict(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2], ), rotation=dict(x=0, y=camera_rotation, z=0))
            midpoint_position_rotation = dict(position=dict(x=midpoint[0], y=midpoint[1], z=midpoint[2]), rotation=dict(x=0,y=0,z=0))
            midpoint_agent_coord = convert_world_to_agent_coordinate(midpoint_position_rotation, agent_state)
            distance_in_agent_coord = midpoint_agent_coord['position']

            agent_centric_middle_of_object = torch.Tensor([distance_in_agent_coord['x'], distance_in_agent_coord['y'], distance_in_agent_coord['z']])

            # Removing this hurts the performance
            agent_centric_middle_of_object = agent_centric_middle_of_object.abs()


            return agent_centric_middle_of_object


def calc_world_coordinates(min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, device, depth_frame):
    with torch.no_grad():
        camera_xyz = (
            torch.from_numpy(camera_xyz - min_xyz).float().to(device)
        )

        depth_frame = torch.from_numpy(depth_frame).to(device)
        depth_frame[depth_frame == -1] = np.NaN
        world_space_point_cloud = depth_frame_to_world_space_xyz(
            depth_frame=depth_frame,
            camera_world_xyz=camera_xyz,
            rotation=camera_rotation,
            horizon=camera_horizon,
            fov=fov,
        )
        return world_space_point_cloud



class PredictionObjectMask(Sensor):
    def __init__(self, type: str,object_query_sensor, rgb_for_detection_sensor,  uuid: str = "predict_object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.type = type
        self.object_query_sensor = object_query_sensor
        self.rgb_for_detection_sensor = rgb_for_detection_sensor
        uuid = '{}_{}'.format(uuid, type)
        self.device = torch.device("cpu")
        self.detection_model = None
        super().__init__(**prepare_locals_for_super(locals()))

    def load_detection_weights(self):
        self.detection_model = ConditionalDetectionModel()
        detection_weight_dir = 'pretrained_models/saved_checkpoints/conditional_segmentation.pytar'
        detection_weight_dict = torch.load(detection_weight_dir, map_location='cpu')
        detection_state_dict = self.detection_model.state_dict()
        for key in detection_state_dict:
            param = detection_weight_dict[key]
            detection_state_dict[key].copy_(param)
        self.detection_model.eval()
        self.detection_model.to(self.device) # do i need to assign this

    def get_detection_masks(self, query_images, images):
        query_images = query_images.to(self.device)
        images = images.to(self.device)
        with torch.no_grad():
            batch, c, w, h = images.shape
            predictions = self.detection_model(dict(rgb=images, target_cropped_object=query_images))
            probs_mask = predictions['object_mask']
            mask = probs_mask.argmax(dim=1).float().unsqueeze(1)#To add the channel back in the end of the image
            return mask

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        if self.detection_model is None:
            self.load_detection_weights()
        query_object = self.object_query_sensor.get_observation(env, task, *args, **kwargs)
        rgb_frame = self.rgb_for_detection_sensor.get_observation(env, task, *args, **kwargs)
        rgb_frame = torch.Tensor(rgb_frame).permute(2, 0, 1)

        predicted_masks = self.get_detection_masks(query_object.unsqueeze(0), rgb_frame.unsqueeze(0)).squeeze(0)

        return predicted_masks.permute(1, 2, 0).cpu() #Channel last





class RealPointNavSensor(Sensor):

    def __init__(self, type: str, noise=0, uuid: str = "point_nav_real", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.noise = noise
        uuid = '{}_{}'.format(uuid, type)
        self.noise_mode = ControllerNoiseModel(
            linear_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.074, 0.036], [0.019, 0.033]),
                _TruncatedMultivariateGaussian([0.189], [0.038]),
            ),
            rotational_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.002, 0.003], [0.0, 0.002]),
                _TruncatedMultivariateGaussian([0.219], [0.019]),
            ),
        )


        super().__init__(**prepare_locals_for_super(locals()))

    def get_accurate_locations(self, env):
        metadata = copy.deepcopy(env.controller.last_event.metadata['agent'])
        return metadata

    def add_translation_noise(self, change_in_xyz, prev_location, real_rotation, belief_rotation):

        if np.abs(change_in_xyz).sum() > 0:
            noise_value_x, noise_value_z = self.noise_mode.linear_motion.linear.sample() * 0.01 * self.noise #to convert to meters
            new_change_in_xyz = change_in_xyz.clone()
            new_change_in_xyz[0] += noise_value_x
            new_change_in_xyz[2] += noise_value_z
            diff_in_rotation = math.radians(belief_rotation - real_rotation)
            # 𝑥2=cos𝛽𝑥1−sin𝛽𝑦1
            # 𝑦2=sin𝛽𝑥1+cos𝛽𝑦1
            new_location = prev_location.clone()
            x = math.cos(diff_in_rotation) * new_change_in_xyz[0] - math.sin(diff_in_rotation) * new_change_in_xyz[2]
            z = math.sin(diff_in_rotation) * new_change_in_xyz[0] + math.cos(diff_in_rotation) * new_change_in_xyz[2]
            new_location[0] += x
            new_location[2] += z
        else:
            new_location = prev_location + change_in_xyz
        return new_location
    def rotate_x_z_around_center(self, x, z, rotation):

        new_x = math.cos(rotation) * x - math.sin(rotation) * z
        new_z = math.sin(rotation) * x + math.cos(rotation) * z

        return new_x, new_z
    def add_rotation_noise(self, change_in_rotation, prev_rotation):
        new_rotation = prev_rotation + change_in_rotation

        if change_in_rotation > 0:
            noise_in_rotation = self.noise_mode.rotational_motion.rotation.sample().item()
            new_rotation += noise_in_rotation
        return new_rotation


    def get_agent_localizations(self, env):

        if self.noise == 0:
            real_current_location = self.get_accurate_locations(env)
            self.real_prev_location = copy.deepcopy(real_current_location)
            self.belief_prev_location = copy.deepcopy(real_current_location)
            return real_current_location
        else:
            real_current_location = self.get_accurate_locations(env)

            if self.real_prev_location is None:
                self.real_prev_location = copy.deepcopy(real_current_location)
                self.belief_prev_location = copy.deepcopy(real_current_location)
            else:
                change_in_xyz = tensor_from_dict(real_current_location['position']) - tensor_from_dict(self.real_prev_location['position'])

                last_step_real_rotation, last_step_belief_rotation = self.real_prev_location['rotation']['y'], self.belief_prev_location['rotation']['y']
                change_in_rotation = real_current_location['rotation']['y'] - self.real_prev_location['rotation']['y']
                belief_camera_xyz = self.add_translation_noise(change_in_xyz, tensor_from_dict(self.belief_prev_location['position']), last_step_real_rotation, last_step_belief_rotation)
                belief_camera_rotation = self.add_rotation_noise(change_in_rotation, last_step_belief_rotation)

                self.belief_prev_location = copy.deepcopy(dict(position=dict(x=belief_camera_xyz[0], y=belief_camera_xyz[1],z=belief_camera_xyz[2]), rotation=dict(x=0,y=belief_camera_rotation, z=0)))
                self.real_prev_location = copy.deepcopy(real_current_location)


            return self.belief_prev_location

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
        if task.num_steps_taken() == 0:
            self.real_prev_location = None
            self.belief_prev_location = None
        goal_obj_id = task.task_info[info_to_search]
        real_object_info = env.get_object_by_id(goal_obj_id)
        real_hand_state = env.get_absolute_hand_state()
        self.get_agent_localizations(env)
        real_agent_state = self.real_prev_location
        belief_agent_state = self.belief_prev_location
        relative_goal_obj = convert_world_to_agent_coordinate(
            real_object_info, belief_agent_state
        )
        relative_hand_state = convert_world_to_agent_coordinate(
            real_hand_state, real_agent_state
        )
        relative_distance = diff_position(relative_goal_obj, relative_hand_state)
        result = convert_state_to_tensor(dict(position=relative_distance))

        return result


class AgentRelativeLocationSensor(Sensor):

    def __init__(self, noise = 0, uuid: str = "agent_relative_location", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.noise = noise
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        agent_initial_state = task.task_info['agent_initial_state']
        current_agent_state = env.controller.last_event.metadata["agent"]

        assert self.noise == 0

        # To avoid gimbal lock
        def is_close_enough(agent_initial_state, current_agent_state, thr = 0.001):
            initial = [agent_initial_state['position'][k] for k in ['x','y','z']] +[agent_initial_state['rotation'][k] for k in ['x','y','z']]
            current = [current_agent_state['position'][k] for k in ['x','y','z']] +[current_agent_state['rotation'][k] for k in ['x','y','z']]
            for i in range(len(initial)):
                if abs(initial[i] - current[i]) > thr:
                    return False
            return True

        if is_close_enough(agent_initial_state, current_agent_state):
            relative_agent_state = {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}
        else:
            relative_agent_state = convert_world_to_agent_coordinate(current_agent_state, agent_initial_state)

        result = convert_state_to_tensor(relative_agent_state)

        return result


class NoisyDepthSensorThor(
    DepthSensorThor
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """
    @lazy_property
    def noise_model(self):
        return RedwoodDepthNoise()

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:
        depth = (env.controller.last_event.depth_frame.copy())
        noisy_depth = self.noise_model.add_noise(depth, depth_normalizer=50)
        return noisy_depth


class MisDetectionNoisyObjectMask(Sensor):
    def __init__(self, type: str,noise, height, width,  uuid: str = "object_mask", distance_thr: float = -1, misdetection_percent = 1, **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.type = type
        self.height = height
        self.width = width
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.misdetection_percent = misdetection_percent
        self.distance_thr = distance_thr
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
        else:
            raise Exception('Not implemented', self.type)

        target_object_id = task.task_info[info_to_search]
        all_visible_masks = env.controller.last_event.instance_masks
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]

            if self.distance_thr > 0:

                agent_location = env.get_agent_location()
                object_location = env.get_object_by_id(target_object_id)['position']
                current_agent_distance_to_obj = sum([(object_location[k] - agent_location[k])**2 for k in ['x', 'z']]) ** 0.5
                if current_agent_distance_to_obj > self.distance_thr or mask_frame.sum() < 20:
                    mask_frame[:] = 0

        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
        if len(env.controller.last_event.instance_masks) == 0:
            fake_mask = np.zeros(env.controller.last_event.frame[:,:,0].shape)
        else:
            fake_mask = random.choice([v for v in env.controller.last_event.instance_masks.values()])
        fake_mask = (np.expand_dims(fake_mask.astype(np.float),axis=-1))
        current_shape = fake_mask.shape
        if random.random() < self.misdetection_percent and result.sum() > 0:
            result = fake_mask
        if (current_shape[0], current_shape[1]) == (self.width, self.height):
            resized_mask = result
        else:
            resized_mask = cv2.resize(result, (self.height, self.width)).reshape(self.width, self.height, 1)

        return resized_mask


class MaskCutoffNoisyObjectMask(Sensor):
    def __init__(self, type: str, mask_cutoff_percent,noise, height, width,  uuid: str = "object_mask", distance_thr: float = -1, **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.type = type
        self.height = height
        self.width = width
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.mask_cutoff_percent = mask_cutoff_percent
        self.distance_thr = distance_thr
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
        else:
            raise Exception('Not implemented', self.type)

        target_object_id = task.task_info[info_to_search]
        all_visible_masks = env.controller.last_event.instance_masks
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]

            if self.distance_thr > 0:

                agent_location = env.get_agent_location()
                object_location = env.get_object_by_id(target_object_id)['position']
                current_agent_distance_to_obj = sum([(object_location[k] - agent_location[k])**2 for k in ['x', 'z']]) ** 0.5
                if current_agent_distance_to_obj > self.distance_thr or mask_frame.sum() < 20:
                    mask_frame[:] = 0

        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
        current_shape = result.shape

        if result.sum() > 0:
            w,h,d = current_shape
            mask = np.random.rand(w,h,d)
            mask = mask < self.mask_cutoff_percent
            mask = mask & (result == 1)
            result[mask] = 0

        if (current_shape[0], current_shape[1]) == (self.width, self.height):
            resized_mask = result
        else:
            resized_mask = cv2.resize(result, (self.height, self.width)).reshape(self.width, self.height, 1) # my gut says this is gonna be slow

        return resized_mask