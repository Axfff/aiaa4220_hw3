#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from gym import spaces

from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.dataset import Dataset, Episode

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface
from habitat.tasks.nav.nav import PointGoalSensor, Success
from hydra.core.config_store import ConfigStore
import habitat_sim
from habitat.tasks.rearrange.rearrange_sensors import NumStepsMeasure
from dataclasses import dataclass
from habitat.config.default_structured_configs import MeasurementConfig

from habitat.tasks.rearrange.utils import rearrange_collision
from habitat.core.embodied_task import Measure
from habitat.tasks.rearrange.social_nav.utils import (
    robot_human_vec_dot_product,
)
from habitat.tasks.nav.nav import DistanceToGoalReward, DistanceToGoal
from habitat.tasks.rearrange.utils import coll_name_matches
try:
    import magnum as mn
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_measure
class DidMultiAgentsCollide(Measure):
    """
    Detects if the multi-agent ( more than 1 humanoids agents) in the scene 
    are colliding with each other at the current step. 
    """

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "did_multi_agents_collide"

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        sim = task._sim
        human_num = task._human_num
        sim.perform_discrete_collision_detection()
        contact_points = sim.get_physics_contact_points()
        found_contact = False

        agent_ids = [
            articulated_agent.sim_obj.object_id
            for articulated_agent in sim.agents_mgr.articulated_agents_iter
        ]
        main_agent_id = agent_ids[0]
        other_agent_ids = set(agent_ids[1:human_num+1])  
        for cp in contact_points:
            if coll_name_matches(cp, main_agent_id):
                if any(coll_name_matches(cp, agent_id) for agent_id in other_agent_ids):
                    found_contact = True
                    break  

        self._metric = found_contact

@registry.register_measure
class HumanCollision(Measure):

    cls_uuid: str = "human_collision"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._ever_collide = False
        super().__init__()

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [DidMultiAgentsCollide._get_uuid()]
        )
        self._metric = 0.0
        self._ever_collide = False

    def update_metric(self, *args, episode, task, observations, **kwargs):
        collid = task.measurements.measures[DidMultiAgentsCollide._get_uuid()].get_metric()
        if collid or self._ever_collide:
            self._metric = 1.0
            self._ever_collide = True
            task.should_end = True
        else:
            self._metric = 0.0

@registry.register_measure
class STL(Measure):
    r"""Success weighted by Completion Time
    """
    cls_uuid: str = "stl"
    
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid, NumStepsMeasure.cls_uuid]
        )

        self._num_steps_taken = 0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, observations=observations, *args, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric() 
        self._num_steps_taken = task.measurements.measures[NumStepsMeasure.cls_uuid].get_metric()

        oracle_time = (
            self._start_end_episode_distance / (0.25 / 10)
        )
        oracle_time = max(oracle_time, 1e-6)
        agent_time = max(self._num_steps_taken, 1e-6)
        self._metric = ep_success * (oracle_time / max(oracle_time, agent_time))

@registry.register_measure
class PersonalSpaceCompliance(Measure):

    cls_uuid: str = "psc"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._use_geo_distance = config.use_geo_distance
        super().__init__()
        
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [NumStepsMeasure.cls_uuid]
        )
        self._compliant_steps = 0
        self._num_steps = 0

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        if self._human_nums == 0:
            self._metric = 1.0
        else:
            robot_pos = self._sim.get_agent_state(0).position
            self._num_steps = task.measurements.measures[NumStepsMeasure.cls_uuid].get_metric()
            compliance = True
            for i in range(self._human_nums):
                human_position = self._sim.get_agent_state(i+1).position

                if self._use_geo_distance:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = robot_pos
                    path.requested_end = human_position
                    found_path = self._sim.pathfinder.find_path(path)

                    if found_path:
                        distance = self._sim.geodesic_distance(robot_pos, human_position)
                    else:
                        distance = np.linalg.norm(human_position - robot_pos, ord=2)
                else:
                    distance = np.linalg.norm(human_position - robot_pos, ord=2)

                if distance < 1.0:
                    compliance = False
                    break                    

            if compliance:
                self._compliant_steps += 1
            self._metric = (self._compliant_steps / self._num_steps)

@registry.register_measure
class MultiAgentNavReward(Measure):
    """
    Reward that gives a continuous reward for the social navigation task.
    """

    cls_uuid: str = "multi_agent_nav_reward"
        
    # @staticmethod
    # def _get_uuid(*args, **kwargs):
    #     return MultiAgentNavReward.cls_uuid
    def _get_uuid(self,*args, **kwargs):
        return self.cls_uuid

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric = 0.0
        config = kwargs["config"]
        # Get the config and setup the hyperparameters
        self._config = config
        self._sim = kwargs["sim"]

        self._use_geo_distance = config.use_geo_distance
        self._allow_distance = config.allow_distance
        self._collide_scene_penalty = config.collide_scene_penalty
        self._collide_human_penalty = config.collide_human_penalty
        self._trajectory_cover_penalty = config.trajectory_cover_penalty
        self._threshold_squared = config.cover_future_dis_thre ** 2
        self._robot_idx = config.robot_idx
        self._close_to_human_penalty = config.close_to_human_penalty
        self._facing_human_dis = config.facing_human_dis

        # FIX: Add hesitation penalty to encourage movement
        self._hesitation_penalty = getattr(config, 'hesitation_penalty', -0.01)
        # Elliptical penalty zone parameters
        self._ellipse_forward_expansion = getattr(config, 'ellipse_forward_expansion', 2.0)
        self._ellipse_backward_shrink = getattr(config, 'ellipse_backward_shrink', 0.5)
        self._ellipse_velocity_threshold = getattr(config, 'ellipse_velocity_threshold', 0.1)
        self._ellipse_velocity_scale = getattr(config, 'ellipse_velocity_scale', 1.0)

        # Obstacle proximity penalty parameters
        self._obstacle_proximity_penalty = getattr(config, 'obstacle_proximity_penalty', -0.0015)
        self._obstacle_proximity_threshold = getattr(config, 'obstacle_proximity_threshold', 1.0)

        self._human_nums = 0
        self._prev_robot_pos = None

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        if "human_num" in episode.info:
            self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        else:
            self._human_nums = 0
        self._metric = 0.0
        self._prev_robot_pos = None
        
    def _check_human_facing_robot(self, human_pos, robot_pos, human_idx):
        base_T = self._sim.get_agent_data(
            human_idx
        ).articulated_agent.sim_obj.transformation
        facing = (
            robot_human_vec_dot_product(human_pos, robot_pos, base_T)
            > self._config.human_face_robot_threshold
        )
        return facing

    def _calculate_elliptical_distance(self, robot_pos, human_pos, human_velocity, human_idx):
        """
        Calculate effective distance from robot to human accounting for elliptical
        penalty zone that expands in the direction of human movement.

        Args:
            robot_pos: Robot position (x, y, z)
            human_pos: Human position (x, y, z)
            human_velocity: Human velocity (linear, angular) from HumanVelocitySensor
            human_idx: Index of the human agent (1-indexed, 0 is robot)

        Returns:
            Effective elliptical distance for penalty calculation
        """
        # Extract 2D positions (x, z plane - ignoring y/height)
        robot_2d = np.array([robot_pos[0], robot_pos[2]])
        human_2d = np.array([human_pos[0], human_pos[2]])

        # Get linear velocity magnitude (first component of human_velocity)
        velocity_magnitude = abs(human_velocity[0]) if len(human_velocity) > 0 else 0.0

        # If velocity is below threshold, use circular distance
        if velocity_magnitude < self._ellipse_velocity_threshold:
            return np.linalg.norm(robot_pos - human_pos, ord=2)

        # Calculate vector from human to robot
        human_to_robot = robot_2d - human_2d
        euclidean_distance = np.linalg.norm(human_to_robot)

        # If too close, return euclidean distance (avoid numerical issues)
        if euclidean_distance < 1e-6:
            return 0.0

        # Get human's facing direction from the articulated agent
        # We use the human's current orientation to determine forward direction
        human_agent_data = self._sim.get_agent_data(human_idx)
        if hasattr(human_agent_data, 'articulated_agent'):
            base_T = human_agent_data.articulated_agent.sim_obj.transformation
            # Extract forward direction (first column of rotation matrix, projected to xz plane)
            forward_3d = base_T.transform_vector(np.array([0, 0, -1]))  # Forward in agent's frame
            forward_2d = np.array([forward_3d[0], forward_3d[2]])
            forward_2d = forward_2d / (np.linalg.norm(forward_2d) + 1e-6)
        else:
            # Fallback: use normalized human_to_robot as reference
            forward_2d = human_to_robot / (euclidean_distance + 1e-6)

        # Calculate ellipse radii based on velocity
        base_radius = self._facing_human_dis

        # Forward direction: expand based on velocity
        forward_expansion = 1.0 + velocity_magnitude * self._ellipse_forward_expansion * self._ellipse_velocity_scale
        r_forward = base_radius * forward_expansion

        # Backward direction: shrink based on velocity
        backward_shrink = 1.0 - velocity_magnitude * self._ellipse_backward_shrink * self._ellipse_velocity_scale
        backward_shrink = max(backward_shrink, 0.3)  # Minimum 30% of base radius
        r_backward = base_radius * backward_shrink

        # Side direction: keep base radius
        r_side = base_radius

        # Project robot position onto velocity-aligned coordinate system
        # Forward axis: along human's movement direction
        # Side axis: perpendicular to movement direction
        forward_component = np.dot(human_to_robot, forward_2d)
        side_component = np.abs(np.dot(human_to_robot, np.array([-forward_2d[1], forward_2d[0]])))

        # Determine which radius to use for forward/backward
        if forward_component >= 0:
            # Robot is in front of human
            r_longitudinal = r_forward
        else:
            # Robot is behind human
            r_longitudinal = r_backward
            forward_component = abs(forward_component)

        # Calculate elliptical distance using the ellipse equation
        # Normalize by respective radii and compute distance from ellipse boundary
        normalized_forward = forward_component / r_longitudinal
        normalized_side = side_component / r_side

        # Distance from ellipse boundary
        # If point is on ellipse: normalized_forward^2 + normalized_side^2 = 1
        # We want the actual distance, scaled by the ellipse
        ellipse_factor = np.sqrt(normalized_forward**2 + normalized_side**2)

        if ellipse_factor < 1e-6:
            return 0.0

        # Effective distance: scale euclidean distance by ellipse factor
        # ellipse_factor > 1 means outside ellipse, < 1 means inside
        effective_distance = euclidean_distance / ellipse_factor

        return effective_distance

    def update_metric(self, *args, episode, task, observations, **kwargs):

        # Start social nav reward
        social_nav_reward = 0.0

        # Component 1: Goal distance reward (strengthened by multiplying by 1.5)
        distance_to_goal_reward = task.measurements.measures[
            DistanceToGoalReward.cls_uuid
        ].get_metric()
        social_nav_reward += 1.5 * distance_to_goal_reward  # Slightly reduced reward multiplier

        # Component 2: Penalize being too close to humans
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        use_k_robot = f"agent_{self._robot_idx}_localization_sensor"
        robot_pos = np.array(observations[use_k_robot][:3])

        if distance_to_target > self._allow_distance:
            # Check if HumanVelocitySensor is available in observations
            has_velocity_sensor = "human_velocity_sensor" in observations

            # Calculate distance to each human and apply proximity penalty
            for i in range(self._human_nums):
                use_k_human = f"agent_{i+1}_localization_sensor"
                human_position = observations[use_k_human][:3]

                # Get human velocity if available
                if has_velocity_sensor:
                    # HumanVelocitySensor shape: (max_humans, 6) where each row is [x, y, z, rot, lin_vel, ang_vel]
                    human_velocity = observations["human_velocity_sensor"][i][4:6]  # Extract [lin_vel, ang_vel]

                    # Use elliptical distance calculation
                    distance = self._calculate_elliptical_distance(
                        robot_pos,
                        human_position,
                        human_velocity,
                        human_idx=i+1  # Human indices start at 1 (0 is robot)
                    )
                else:
                    # Fallback to original distance calculation if velocity sensor not available
                    if self._use_geo_distance:
                        path = habitat_sim.ShortestPath()
                        path.requested_start = robot_pos
                        path.requested_end = human_position
                        found_path = self._sim.pathfinder.find_path(path)
                        if found_path:
                            distance = self._sim.geodesic_distance(robot_pos, human_position)
                        else:
                            distance = np.linalg.norm(human_position - robot_pos, ord=2)
                    else:
                        distance = np.linalg.norm(human_position - robot_pos, ord=2)

                # Apply penalty if within threshold
                if distance < self._facing_human_dis:
                    penalty = self._close_to_human_penalty * np.exp(-distance / self._facing_human_dis)
                    social_nav_reward += penalty

        # Component 3: Collision detection for two agents
        did_agents_collide = task.measurements.measures[
            DidMultiAgentsCollide._get_uuid()
        ].get_metric()
        if did_agents_collide:
            task.should_end = True
            social_nav_reward += self._collide_human_penalty

        # Component 4: Collision detection for the main agent and the scene
        did_rearrange_collide, collision_detail = rearrange_collision(
            self._sim, True, ignore_base=False, agent_idx=self._robot_idx
        )
        if did_rearrange_collide:
            social_nav_reward += self._collide_scene_penalty

        # Component 4.5: Soft penalty for approaching static obstacles (depth-based)
        # This provides gentle guidance away from walls/obstacles before collision
        depth_sensor_key = f"agent_{self._robot_idx}_articulated_agent_jaw_depth"
        if depth_sensor_key in observations:
            # Get depth observation (shape: [HEIGHT, WIDTH, 1])
            depth_obs = observations[depth_sensor_key]

            # Find minimum depth value (closest obstacle in any direction)
            # Depth values are typically normalized [0, 1] or in meters [0, max_depth]
            # We need to handle both cases
            min_depth = np.min(depth_obs)

            # Check if depth is normalized (values between 0 and 1)
            # If normalized, we need to denormalize using max_depth (typically 10.0m)
            max_depth = 10.0  # Standard max depth for Habitat depth sensors
            if min_depth <= 1.0:
                # Likely normalized, denormalize to meters
                min_depth = min_depth * max_depth

            # Apply soft penalty if obstacle is within threshold distance
            if min_depth < self._obstacle_proximity_threshold:
                # Exponential decay penalty - same pattern as human proximity
                obstacle_penalty = self._obstacle_proximity_penalty * np.exp(
                    -min_depth / self._obstacle_proximity_threshold
                )
                social_nav_reward += obstacle_penalty

        # Component 5: Trajectory overlap penalty with time-based weighting
        if distance_to_target > self._allow_distance and "human_future_trajectory" in task.measurements.measures:
            human_future_trajectory_temp = task.measurements.measures['human_future_trajectory']._metric
            for trajectory in human_future_trajectory_temp.values():
                for t, point in enumerate(trajectory):
                    time_weight = 1.0 / (1 + t)  # Time-weighted penalty
                    if np.sum((robot_pos - point) ** 2) < self._threshold_squared:
                        social_nav_reward += self._trajectory_cover_penalty * time_weight
                        break

        # FIX Component 6: Hesitation penalty - penalize robot for not moving
        # This prevents the "freeze" strategy where robot stops immediately
        if self._prev_robot_pos is not None and distance_to_target > self._allow_distance:
            movement = np.linalg.norm(robot_pos - self._prev_robot_pos)
            # If robot barely moved (< 0.05m), apply penalty
            if movement < 0.05:
                social_nav_reward += self._hesitation_penalty

        self._prev_robot_pos = robot_pos.copy()

        self._metric = social_nav_reward

@registry.register_measure
class HumanVelocityMeasure(UsesArticulatedAgentInterface, Measure):
    """
    The measure for ORCA
    """

    cls_uuid: str = "human_velocity_measure"

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim
        self.human_num = kwargs['task']._human_num
        self.velo_coff = np.array([[0, 1]] * 6)
        self.velo_base = np.array([[0.25, np.deg2rad(10)]] * 6)
        
        super().__init__(*args, sim=sim, **kwargs)
        self._metric = self.velo_base * self.velo_coff 

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HumanVelocityMeasure.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.human_num = task._human_num
        self.velo_coff = np.array([[0.0, 0.0]] * 6)
        self.velo_base = np.array([[0.25, np.deg2rad(10)]] * 6)
        self._metric = self.velo_base * self.velo_coff 

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = self.velo_base * self.velo_coff 

def merge_paths(paths):
    merged_path = []
    for i, path in enumerate(paths):
        if i > 0:
            path = path[1:]
        merged_path.extend(path)
    return merged_path


@registry.register_measure
class HumanFutureTrajectory(UsesArticulatedAgentInterface, Measure):
    """
    The measure for future prediction of social crowd navigation
    """

    cls_uuid: str = "human_future_trajectory"

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim
        self.num_agents = sim.num_articulated_agents
        self.target_dict = [[[0, 0, 0]] for _ in range(self.num_agents-1)]
        self.path_dict = {}
        super().__init__(*args, sim=sim, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HumanFutureTrajectory.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def _path_to_point(self, point_a,point_b):

        path = habitat_sim.ShortestPath()
        path.requested_start = point_a 
        path.requested_end = point_b
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [point_a, point_b]
        return path.points

    def update_metric(self, *args, episode, task, observations, **kwargs):
        for agent_idx, target in enumerate(self.target_dict):
            path = []
            
            agent_pos = self._sim.get_agent_data(agent_idx+1).articulated_agent.base_pos
            for i in range(-1,len(target)):
                if i == -1:
                    path_point = np.array(agent_pos)
                else:
                    path_point = target[i]

                if i >= 0:
                    temp_path = self._path_to_point(prev_point, path_point)
                    path.append(temp_path)
                
                prev_point = path_point

            if path == []:
                self.path_dict[agent_idx + 1] = []
            else:
                temp_merged_path = merge_paths(path)
                output_length = min(5, len(temp_merged_path))
                self.path_dict[agent_idx + 1] = temp_merged_path[:output_length]

        self._metric = self.path_dict

@registry.register_measure
class HumanFutureTrajectory(UsesArticulatedAgentInterface, Measure):
    """
    The measure for future prediction of social crowd navigation.
    """

    cls_uuid: str = "human_future_trajectory"

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim
        self.human_num = kwargs['task']._human_num
        self.output_length = 5
        self.target_dict = self._initialize_target_dict(self.human_num)
        self.path_dict = {}
        super().__init__(*args, sim=sim, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HumanFutureTrajectory.cls_uuid

    def _initialize_target_dict(self, human_num):
        """Initialize the target dictionary with default values."""
        return np.full((human_num, 2, 3), -100, dtype=np.float32).tolist()

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.human_num = task._human_num
        self.target_dict = self._initialize_target_dict(self.human_num)
        self.path_dict = {}
        self._metric = {}

    def _path_to_point(self, point_a, point_b):
        """Get the shortest path between two points."""
        path = habitat_sim.ShortestPath()  
        path.requested_start = point_a 
        path.requested_end = point_b
        found_path = self._sim.pathfinder.find_path(path)
        return path.points if found_path else [point_a, point_b]

    def _process_path(self, path):
        """Process the path by merging and padding/truncating to the desired length."""
        temp_merged_path = merge_paths(path)
        
        if len(temp_merged_path) < self.output_length:
            padding = np.full((self.output_length - len(temp_merged_path), 3), temp_merged_path[-1], dtype=np.float32)
            temp_merged_path = np.concatenate([temp_merged_path, padding], axis=0)
        else:
            temp_merged_path = np.array(temp_merged_path[:self.output_length], dtype=np.float32)
        
        return temp_merged_path.tolist()

    def update_metric(self, *args, episode, task, observations, **kwargs):
        for agent_idx, target in enumerate(self.target_dict):
            path = []
            agent_pos = np.array(self._sim.get_agent_data(agent_idx + 1).articulated_agent.base_pos)

            prev_point = agent_pos
            for i in range(len(target)):
                path_point = np.array(target[i])
                temp_path = self._path_to_point(prev_point, path_point)
                path.append(temp_path)
                prev_point = path_point

            self.path_dict[agent_idx + 1] = self._process_path(path)
            
        self._metric = self.path_dict

@dataclass
class MultiAgentNavReward(MeasurementConfig):
    r"""
    The reward for the multi agent navigation tasks.
    """
    type: str = "MultiAgentNavReward"

    # If we want to use geo distance to measure the distance
    # between the robot and the human
    use_geo_distance: bool = True
    # discomfort for multi agents
    allow_distance: float = 0.5
    collide_scene_penalty: float = -0.25
    collide_human_penalty: float = -0.5
    facing_human_dis: float = 1.0
    human_face_robot_threshold: float = 0.5
    close_to_human_penalty: float = -0.025
    trajectory_cover_penalty: float = -0.025
    cover_future_dis_thre: float = -0.05
    # FIX: Add hesitation penalty parameter
    hesitation_penalty: float = -0.01
    # Set the id of the agent
    robot_idx: int = 0
    # Elliptical penalty zone parameters (velocity-aware)
    ellipse_forward_expansion: float = 2.0
    ellipse_backward_shrink: float = 0.5
    ellipse_velocity_threshold: float = 0.1
    ellipse_velocity_scale: float = 1.0
    # Obstacle proximity penalty parameters (distance-based soft penalty)
    obstacle_proximity_penalty: float = -0.0015
    obstacle_proximity_threshold: float = 1.0

@dataclass
class DidMultiAgentsCollideConfig(MeasurementConfig):
    type: str = "DidMultiAgentsCollide"
    
@dataclass
class STLMeasurementConfig(MeasurementConfig):
    type: str = "STL"

@dataclass
class PersonalSpaceComplianceMeasurementConfig(MeasurementConfig):
    type: str = "PersonalSpaceCompliance"
    use_geo_distance: bool = True
    
@dataclass
class HumanCollisionMeasurementConfig(MeasurementConfig):
    type: str = "HumanCollision"

@dataclass
class HumanVelocityMeasurementConfig(MeasurementConfig):
    type: str = "HumanVelocityMeasure"

@dataclass
class HumanFutureTrajectoryMeasurementConfig(MeasurementConfig):
    type: str = "HumanFutureTrajectory"


cs = ConfigStore.instance()

cs.store(
    package="habitat.task.measurements.multi_agent_nav_reward",
    group="habitat/task/measurements",
    name="multi_agent_nav_reward",
    node=MultiAgentNavReward,
)
cs.store(
    package="habitat.task.measurements.stl",
    group="habitat/task/measurements",
    name="stl",
    node=STLMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.psc",
    group="habitat/task/measurements",
    name="psc",
    node=PersonalSpaceComplianceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.human_collision",
    group="habitat/task/measurements",
    name="human_collision",
    node=HumanCollisionMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.did_multi_agents_collide",
    group="habitat/task/measurements",
    name="did_multi_agents_collide",
    node=DidMultiAgentsCollideConfig,
)
cs.store(
    package="habitat.task.measurements.human_velocity_measure",
    group="habitat/task/measurements",
    name="human_velocity_measure",
    node=HumanVelocityMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.human_future_trajectory",
    group="habitat/task/measurements",
    name="human_future_trajectory",
    node=HumanFutureTrajectoryMeasurementConfig,
)