import rospy
from ur5_rl.envs.robot_envs import UR5Env
from ur5_rl.envs import JointGroupPublisher
from ur5_rl.envs.task_envs.RLObservation import RLObservation

from geometry_msgs.msg import Pose, Point
from gazebo_msgs.msg import ModelState

import numpy as np


class UR5EnvGoal(UR5Env):
    def __init__(self, controllers_list, link_names, joint_limits, target_limits, pub_topic_name):
        rospy.logdebug("Start UR5EnvGoal INIT...")

        UR5Env.__init__(self, controllers_list=controllers_list,
                              link_names=link_names,
                              joint_limits=joint_limits)

        self._target_limits = target_limits

        self.__target_position = None
        self.__prev_distance = None
        # Create publisher for robot movement
        self.gazebo.unpauseSim()
        self._publisher = JointGroupPublisher(pub_topic_name, self.controllers_object)
        self._set_model_state = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        rate = rospy.Rate(120)
        while (self._set_model_state.get_num_connections() == 0):
            rospy.logdebug(
                "No publishers to /gazebo/set_link_state yet so we wait and try again")
            rate.sleep()
        self.gazebo.pauseSim()

        rospy.logdebug("Finished UR5EnvGoal INIT...")

    def _reset_sim(self):
        """Resets a simulation
        """
        has_collision = True
        while has_collision:
            self._reset_env_state()
            rospy.logdebug("Pausing SIM...")
            self.gazebo.pauseSim()
            rospy.logdebug("Reset SIM...")
            self.gazebo.resetWorld()
            rospy.logdebug("Unpausing Sim...")
            self.gazebo.unpauseSim()
            rospy.logdebug("Reseting Controllers...")
            self.controllers_object.reset_controllers(self.controllers_list)
            rospy.logdebug("Checking Publishers Connections...")
            self._publisher.check_publishers_connection()
            rospy.logdebug("Checking All Systems...")
            self._check_all_systems_ready()
            rospy.logdebug("Checking Init Pose for Target...")
            self._set_init_target_pose()
            rospy.logdebug("Setting Init Pose for Arm...")
            self._set_init_pose(self._joint_limits)
            has_collision = self._has_collision()

    def _set_init_pose(self, joint_limits):
        """Sets the Robot in its init pose
        """
        joints_array = self._generate_random_pose([joint_limits['lower'], joint_limits['upper']])

        self._publisher.move_joints(joints_array)
        rospy.logdebug(f'Moved joints to initial position {joints_array}')

    def _set_init_target_pose(self):
        # assume that cubes are lying on the table and z is unchangable
        target_coords = self._generate_point_in_sphere(self._target_limits['radius'])
        
        target_pose = Pose()
        target_point = Point()
        target_point.x = target_coords[0]
        target_point.y = target_coords[1]
        target_point.z = target_coords[2]
        target_pose.position = target_point
        
        link_state_msg = ModelState(model_name='cube1', pose=target_pose)
        self._set_model_state.publish(link_state_msg)
        rospy.logdebug(f'Moved target to initial position {target_pose.position}')

        self.__target_position = np.array(target_coords)
        self.__target_position[2] += 2 * self._target_limits['target_size'][2]

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        self.__prev_distance = np.linalg.norm(np.array(self._ur5_state.end_effector_position) - self.__target_position)

    def _compute_reward(self, obs, done):
        """Calculates the reward to give based on the observations given.
        """
        # collision_penalty = -100
        success_reward = 100

        distance = np.linalg.norm(np.array(self._ur5_state.end_effector_position) - self.__target_position)
        if distance <= 0.05:
            rospy.logdebug('Reached goal! HOORAY!')
            return success_reward, True, {'distance': 0.0}
        # elif done:  # collision happened
        #     rospy.logwarn('Collision happened, moving on to next episode')
        #     return collision_penalty, done, {'distance': 0.0}
        else:
            if distance <= self.__prev_distance:
                distance_reward = 0
            else:
                distance_reward = 100 * (self.__prev_distance - distance)
            self.__prev_distance = distance
            return distance_reward, done, {'distance': distance}

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        self._publisher.move_joints(action.tolist())

    def _is_done(self):
        """Checks if episode done based on observations given.
        """
        return self._has_collision()

    def _update_robot_state(self, joint_states, eef_position):
        self._ur5_state.update(joint_states, eef_position)

    def _get_obs(self):
        joint_states = self.get_current_joint_states()
        end_effector_position = self.get_current_eef_position()
        return RLObservation(joint_states, end_effector_position,
                             self._joint_limits, self.__target_position).get_model_input()

    def _has_collision(self):
        return self.check_collisions()

    @staticmethod
    def action_dim():
        return 6

    @staticmethod
    def state_dim():
        return RLObservation.dim()

    def _generate_point_in_sphere(self, radius):
        r = radius * np.random.uniform() ** 0.333
        theta = np.random.uniform() * 2 * np.pi
        phi = np.random.uniform() * np.pi

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        # assume robot is located at (0, 0, radius)
        z = radius + r * np.cos(phi)

        return [x, y, z]
