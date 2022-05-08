import rospy
from ur5_rl.envs.robot_envs import UR5Env
from ur5_rl.envs import JointGroupPublisher

from geometry_msgs.msg import Pose, Point
from gazebo_msgs.msg import ModelState

import torch
from torch.distributions.uniform import Uniform
import numpy as np


class UR5EnvGoal(UR5Env):
    def __init__(self, controllers_list, joint_limits, target_limits, pub_topic_name):
        rospy.logdebug("Start UR5EnvGoal INIT...")

        UR5Env.__init__(self, controllers_list=controllers_list,
                              joint_limits=joint_limits,
                              target_limits=target_limits)

        # Create  publisher for robot movement
        self.gazebo.unpauseSim()
        self._publisher = JointGroupPublisher(pub_topic_name, self.controllers_object)
        self.gazebo.pauseSim()

        rospy.logdebug("Finished UR5EnvGoal INIT...")

    def _reset_sim(self):
        """Resets a simulation
        """
        is_collided = True
        while is_collided:
            rospy.logdebug("Pausing SIM...")
            self.gazebo.pauseSim()
            rospy.logdebug("Reset SIM...")
            self.gazebo.resetSimulation()
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
            is_collided = self._set_init_pose(self.joint_limits)
        
    
    def _set_init_pose(self, joint_limits):
        """Sets the Robot in its init pose
        """
        joints_array = self._generate_init_pose(joint_limits) 

        self._publisher.move_joints(joints_array)
        return False #self._is_done()


    def _set_init_target_pose(self):
        # assume that cubes are lying on the table and z is unchangable
        target_coords = self._generate_init_pose([self.target_limits[0][:2], self.target_limits[1][:2]])
        
        target_pose = Pose()
        target_point = Point()
        target_point.x = target_coords[0]
        target_point.y = target_coords[1]
        # assum lower and upper for z are equal
        target_point.z = self.target_limits[0][2]
        target_pose.position = target_point
        
        link_state_msg = ModelState(model_name='cube1', pose=target_pose)
        self._set_model_state.publish(link_state_msg)
        target_coords.append(self.target_limits[0][2])
        self.target_position = target_coords
        self.target_position[2] += 2 * self.target_limits[2][2]

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        self.initial_distance = np.linalg.norm(np.array(self.end_effector_position) - np.array(self.target_position))

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        # TODO depends on is_done
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        self._publisher.move_joints(action)

    def _get_obs(self):
        self.end_effector_position = list(self.get_current_xyz())
        print(type(self.joints_state.position[:6]))
        observations = list(self.joints_state.position[:6]) + \
                       list(self.joints_state.velocity[:6]) + \
                       self.end_effector_position
        return observations

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        #TODO
        raise NotImplementedError()

    def _generate_init_pose(self, limits):
        """ limits[0] - lower limits
            limits[1] - upper limits 
        """
        distrubtion = Uniform(torch.Tensor(limits[0]), 
                                  torch.Tensor(limits[1]))
        return list(distrubtion.sample().detach().cpu().numpy())
