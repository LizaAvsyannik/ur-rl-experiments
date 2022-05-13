import rospy
from ur5_rl.envs.robot_envs import UR5Env
from ur5_rl.envs import JointGroupPublisher

from geometry_msgs.msg import Pose, Point
from gazebo_msgs.msg import ModelState

import torch
from torch.distributions.uniform import Uniform
import numpy as np


class UR5EnvGoal(UR5Env):
    def __init__(self, controllers_list, joint_names, joint_limits, link_names, target_limits, pub_topic_name):
        rospy.logdebug("Start UR5EnvGoal INIT...")

        UR5Env.__init__(self, controllers_list=controllers_list,
                              joint_names=joint_names,
                              joint_limits=joint_limits,
                              link_names=link_names,
                              target_limits=target_limits)


        self.model_name = 'robot'
        self.reference_frame = 'world'

        # Create  publisher for robot movement
        self.gazebo.unpauseSim()
        self._publisher = JointGroupPublisher(pub_topic_name, self.controllers_object)
        self.robot_inital_pose = self.gazebo.get_model_state((self.model_name, self.reference_frame))
        self.gazebo.pauseSim()

        rospy.logdebug("Finished UR5EnvGoal INIT...")

    def _reset_sim(self):
        """Resets a simulation
        """
        is_collided = True
        while is_collided:
            # 1st: reseting the world
            # self.gazebo.resetWorld()
            # self.gazebo.unpauseSim()
            urdf_file = rospy.get_param('/robot_description')

            # 2nd: getting robot's original position\
            rospy.logdebug("Getting original position...")
            
            
            # 3rd: delete the model
            rospy.logdebug("Deleting model...")
            self.gazebo.delete_model((self.model_name))

            # 3rd: spawn the model
            rospy.logdebug("Spawning model...")
            self.gazebo.spawn_urdf_model((self.model_name, urdf_file, '', self.robot_inital_pose.pose, self.reference_frame))
            
            # 4th: set new position
            joints_array = [0.0, -2.33, 1.57, 0.0, 0.0, 0.2]#self._generate_init_pose([self.joint_limits['lower'], self.joint_limits['upper']])
            # print(self.joint_names)
            # self.gazebo.set_model_configuration((model_name, urdf_file, self.joint_names, joints_array))
            # rospy.logdebug("Set new model...")
            is_collided = False
            
            # # 5th: unpausing the simulation and checking for collision
            # self.gazebo.unpauseSim()
            # is_collided = self._is_done()
            # if is_collided: 
            #     continue
            # else: 
            #     rospy.logdebug(f'Moved joints to initial position {joints_array}')

            #     rospy.logdebug("Reseting Controllers...")
            #     self.controllers_object.reset_controllers(self.controllers_list)
            #     rospy.logdebug("Checking Publishers Connections...")
            #     self._publisher.check_publishers_connection()
            #     rospy.logdebug("Checking Init Pose for Target...")
            #     self._set_init_target_pose()
            #     rospy.logdebug("Checking All Systems...")
            #     self._check_all_systems_ready()

    def _set_init_target_pose(self):
        # assume that cubes are lying on the table and z is unchangable
        target_coords = self._generate_init_pose([self.target_limits['lower'][:2], self.target_limits['upper'][:2]])
        
        target_pose = Pose()
        target_point = Point()
        target_point.x = target_coords[0]
        target_point.y = target_coords[1]
        # assum lower and upper for z are equal
        target_point.z = self.target_limits['lower'][2]
        target_pose.position = target_point
        
        link_state_msg = ModelState(model_name='cube1', pose=target_pose)
        self._set_model_state.publish(link_state_msg)
        rospy.logdebug(f'Moved target to initial position {target_pose.position}')

        target_coords.append(self.target_limits['lower'][2])
        self.target_position = target_coords
        self.target_position[2] += 2 * self.target_limits['target_size'][2]

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        self.prev_distance = np.linalg.norm(np.array(self.end_effector_position) - np.array(self.target_position))

    def _compute_reward(self, action, done):
        """Calculates the reward to give based on the observations given.
        """
        effort_penalty = np.linalg.norm(np.array(action))
        collision_penalty = -100
        success_reward = 100

        if not done:
            effort_coeff = 1e-2
            effort_penalty *= effort_coeff
            print(self.end_effector_position)
            distance = np.linalg.norm(np.array(self.end_effector_position) - np.array(self.target_position))
            distance_reward = self.prev_distance - distance
            self.prev_distance = distance
            
            if distance_reward > 0.00:
                rospy.logwarn("INCREASE IN DISTANCE")
            else:
                rospy.loginfo("DECREASE IN DISTANCE")

            if distance <= 0.05:
                rospy.logdebug('Reached goal! HOORAY!')
                return success_reward - effort_penalty, True

            return distance_reward - effort_penalty, done

        else: # collision happened
            rospy.logwarn('Collsion happend, moving to next epsisode')
            return collision_penalty, done

       
    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        position = [sum(x) for x in zip(list(self.joints_state.position[:6]), action)]
        self._publisher.move_joints(position)

    def _get_obs(self):
        self.end_effector_position = list(self.get_current_xyz())
        observations = list(self.joints_state.position[:6]) + \
                       list(self.joints_state.velocity[:6]) + \
                       self.end_effector_position
        return observations

    def _is_done(self):
        """Checks if episode done based on observations given.
        """
        return self.check_current_collisions()

    def _generate_init_pose(self, limits):
        """ limits[0] - lower limits
            limits[1] - upper limits 
        """
        distrubtion = Uniform(torch.Tensor(limits[0]), 
                                  torch.Tensor(limits[1]))
        return list(distrubtion.sample().detach().cpu().numpy())
