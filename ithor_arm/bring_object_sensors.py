"""Utility classes and functions for sensory inputs used by the models."""

import random

import cv2
import gym
import torch

from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor

from ithor_arm.ithor_arm_constants import ALL_POSSIBLE_OBJECTS
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from manipulathor_baselines.bring_object_baselines.models.detection_model import ConditionalDetectionModel
from utils.thor_category_names import thor_possible_objects
from typing import Union, Optional, Any
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment


class ObjectCategorySensor(Sensor):
    def __init__(self, type: str, uuid: str = "category_code", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        uuid = '{}_{}'.format(uuid, type)
        self.type = type
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
        object_type = task.task_info[info_to_search].split('|')[0]
        object_type_categ_ind = ALL_POSSIBLE_OBJECTS.index(object_type)
        return torch.tensor(object_type_categ_ind)



class DepthSensorThor(
    DepthSensor[
        Union[IThorEnvironment],
        Union[Task[IThorEnvironment]],
    ]
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        depth = (env.controller.last_event.depth_frame.copy())

        return depth


class PickedUpObjSensor(Sensor):
    def __init__(self, uuid: str = "pickedup_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return task.object_picked_up

class RelativeArmDistanceToGoal(Sensor):
    def __init__(self, uuid: str = "relative_arm_dist", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))
    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        is_object_picked_up = task.object_picked_up
        if not is_object_picked_up:
            distance = task.arm_distance_from_obj()
        else:
            distance = task.obj_distance_from_goal()
        return distance


class PreviousActionTaken(Sensor):
    def __init__(self,  uuid: str = "previous_action_taken", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))
    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        last_action = task._last_action_str
        action_list = task._actions
        result = torch.zeros(len(action_list))
        if last_action != None:
            result[action_list.index(last_action)] = 1
        return result.bool()


class IsGoalObjectVisible(Sensor):
    def __init__(self,  uuid: str = "is_goal_object_visible", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))
    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        if not task.object_picked_up:
            result = task.source_observed_reward
        else:
            result = task.goal_observed_reward
        return torch.tensor(result).bool()


class NoGripperRGBSensorThor(RGBSensorThor):
    def frame_from_env(
            self, env: IThorEnvironment, task: Task[IThorEnvironment]
    ) -> np.ndarray:  # type:ignore
        env.controller.step('ToggleMagnetVisibility')
        frame = env.current_frame.copy()
        env.controller.step('ToggleMagnetVisibility')
        return frame

class CategorySampleSensor(Sensor):
    def __init__(self, type: str, uuid: str = "category_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        uuid = '{}_{}'.format(uuid, type)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        feature_name = 'object_query'
        if self.type == 'source':
            info_to_search = 'source_' + feature_name
        elif self.type == 'destination':
            info_to_search = 'goal_' + feature_name
        else:
            raise Exception('Not implemented', self.type)
        image = task.task_info[info_to_search]
        return image


class CategoryFeatureSampleSensor(Sensor):
    def __init__(self, type: str, uuid: str = "category_object_feature", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.type = type
        uuid = '{}_{}'.format(uuid, type)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        feature_name = 'object_query_feature'
        if self.type == 'source':
            info_to_search = 'source_' + feature_name
        elif self.type == 'destination':
            info_to_search = 'goal_' + feature_name
        else:
            raise Exception('Not implemented', self.type)
        feature = task.task_info[info_to_search]
        return feature



class NoisyObjectMask(Sensor):
    def __init__(self, type: str,noise, height, width,  uuid: str = "object_mask", distance_thr: float = -1, recall_percent = 1, **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.type = type
        self.height = height
        self.width = width
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.recall_percent = recall_percent
        self.distance_thr = distance_thr
        super().__init__(**prepare_locals_for_super(locals()))
        assert self.recall_percent == 1 or self.noise == 0

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
        fake_mask, is_real_mask = add_mask_noise(result, fake_mask, noise=self.noise)
        current_shape = fake_mask.shape
        if (current_shape[0], current_shape[1]) == (self.width, self.height):
            resized_mask = fake_mask
        else:
            resized_mask = cv2.resize(fake_mask, (self.height, self.width)).reshape(self.width, self.height, 1) # my gut says this is gonna be slow
        if self.recall_percent < 1:
            if random.random() > self.recall_percent:
                resized_mask[:] = 0
        return resized_mask

class NoMaskSensor(NoisyObjectMask):
    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
        result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
        return result

class NoisyObjectRegion(NoisyObjectMask):
    def __init__(self, type: str,noise, region_size,height, width,  uuid: str = "object_mask", distance_thr: float = -1, **kwargs: Any):
        super().__init__(**prepare_locals_for_super(locals()))
        self.region_size = region_size
        assert self.region_size == 14, 'the following need to be changed'

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        mask = super(type(self), self).get_observation(env, task, *args, **kwargs)


        region = cv2.resize(mask, (self.region_size, self.region_size))
        region = (region > 0.1).astype(float).reshape(self.region_size, self.region_size, 1)
        assert self.region_size == 14, 'the folliowing number wont work'
        number_of_repeat = 16
        region = region.repeat(number_of_repeat, axis=0).repeat(number_of_repeat, axis=1)


        return region


def add_mask_noise(real_mask, fake_mask, noise):
    TURN_OFF_RATE = noise
    REMOVE_RATE = noise
    REPLACE_WITH_FAKE = noise

    result = real_mask.copy()

    random_prob = random.random()
    if random_prob < REMOVE_RATE:
        result[:] = 0.
        is_real_mask = False
    elif random_prob < REMOVE_RATE + REPLACE_WITH_FAKE:
        result = fake_mask
        is_real_mask = False
    elif random_prob < REMOVE_RATE + REPLACE_WITH_FAKE + TURN_OFF_RATE:
        w, h, d = result.shape
        mask = np.random.rand(w, h, d)
        mask = mask < TURN_OFF_RATE
        mask = mask & (result == 1)
        result[mask] = 0
        is_real_mask = True
    else:
        is_real_mask = True

    masks_are_changed_but_still_similar = (result != real_mask).sum() == 0 and not is_real_mask
    if masks_are_changed_but_still_similar:
        is_real_mask = True

    return result, is_real_mask




class DestinationObjectSensor(Sensor):
    def __init__(self, uuid: str = "destination_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return task.task_info['goal_object_id'].split('|')[0]
class TargetObjectMask(Sensor):
    def __init__(self, uuid: str = "target_object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['source_object_id']
        all_visible_masks = env.controller.last_event.instance_masks
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]
        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        return (np.expand_dims(mask_frame.astype(np.float),axis=-1))



class TargetObjectType(Sensor):
    def __init__(self, uuid: str = "target_object_type", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['source_object_id']
        target_object_type = target_object_id.split('|')[0]
        return thor_possible_objects.index(target_object_type)

class RawRGBSensorThor(Sensor):
    def __init__(self, uuid: str = "raw_rgb_lowres", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return env.current_frame.copy()



class TargetLocationMask(Sensor): #
    def __init__(self, uuid: str = "target_location_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['goal_object_id']
        all_visible_masks = env.controller.last_event.instance_masks
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]
        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        return (np.expand_dims(mask_frame.astype(np.float),axis=-1))

class TargetLocationType(Sensor): #
    def __init__(self, uuid: str = "target_location_type", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['goal_object_id']
        target_location_type = target_object_id.split('|')[0]
        return thor_possible_objects.index(target_location_type)


class TargetLocationBBox(Sensor): #
    def __init__(self, uuid: str = "target_location_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['goal_object_id']

        all_visible_masks = env.controller.last_event.instance_detections2D
        box_as_mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
        if target_object_id in all_visible_masks:
            x1, y1, x2, y2 = all_visible_masks[target_object_id]
            box_as_mask_frame[y1:y2, x1:x2] = 1.

        return (np.expand_dims(box_as_mask_frame.astype(np.float),axis=-1))


def add_noise(result):
    TURN_OFF_RATE = 0.05
    ADD_PIXEL_RATE = 0.0005
    REMOVE_RATE = 0.1
    if random.random() < REMOVE_RATE:
        result[:] = 0.
    else:
        w, h, d = result.shape
        mask = np.random.rand(w, h, d)
        mask = mask < TURN_OFF_RATE
        mask = mask & (result == 1)
        result[mask] = 0


        mask = np.random.rand(w, h, d)
        mask = mask < ADD_PIXEL_RATE
        mask = mask & (result == 0)
        result[mask] = 1

    return result

class NoisyTargetLocationBBox(Sensor): #
    def __init__(self, uuid: str = "target_location_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['goal_object_id']

        all_visible_masks = env.controller.last_event.instance_detections2D
        box_as_mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
        if target_object_id in all_visible_masks:
            x1, y1, x2, y2 = all_visible_masks[target_object_id]
            box_as_mask_frame[y1:y2, x1:x2] = 1.
        result = (np.expand_dims(box_as_mask_frame.astype(np.float),axis=-1))

        return add_noise(result)



class NoisyTargetObjectBBox(Sensor):
    def __init__(self, uuid: str = "target_object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['source_object_id']
        all_visible_masks = env.controller.last_event.instance_detections2D
        box_as_mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
        if target_object_id in all_visible_masks:
            x1, y1, x2, y2 = all_visible_masks[target_object_id]
            box_as_mask_frame[y1:y2, x1:x2] = 1.
        result = (np.expand_dims(box_as_mask_frame.astype(np.float),axis=-1))

        return add_noise(result)
