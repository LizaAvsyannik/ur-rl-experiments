import rospy
from ur5_rl.envs import RobotGazeboEnv
from ur5_rl.envs.robot_envs.ur5_state import UR5State
from gazebo_msgs.msg import LinkStates, ContactsState
from sensor_msgs.msg import JointState

import numpy as np


class UR5Env(RobotGazeboEnv):
    def __init__(self, ns, controllers_list, link_names, joint_limits):
        rospy.logdebug("Start UR5Env INIT...")
        RobotGazeboEnv.__init__(self, ns, controllers_list=controllers_list)
        
        self.link_names = link_names
        self._joint_limits = joint_limits
        self._ur5_state = UR5State()

        rospy.logdebug("UR5Env unpause...")
        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers(self.controllers_list)

        # Subscribe link and joint states 
        self._get_link_states = rospy.Subscriber('/' + self.ns + "/gazebo/link_states", LinkStates,
                            self.link_state_callback, queue_size=1)
        self._get_joint_states = rospy.Subscriber('/' + self.ns + "/joint_states", JointState,
                            self.joint_states_callback, queue_size=1)
        self._collision_sensors = [rospy.Subscriber('/' + self.ns + f"/{name}_collision_sensor", ContactsState,
                            self.contact_state_callback, queue_size=1) for name in self.link_names]
        rospy.logdebug("Subscribed to states")
        self._check_all_systems_ready()
        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished UR5Env INIT...")

    def _reset_env_state(self):
        self._ur5_state.reset()

    def _check_all_systems_ready(self):
        """ We check that all systems are ready
        """
        link_states_msg = None
        while link_states_msg is None and not rospy.is_shutdown():
            try:
                link_states_msg = rospy.wait_for_message(
                    '/' + self.ns + "/gazebo/link_states", LinkStates, timeout=0.1)
                self.link_state = link_states_msg
                rospy.logdebug("Current link_states READY")
            except Exception as e:
                rospy.logdebug(
                    "Current links not ready yet, retrying==>"+str(e))

        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message(
                    '/' + self.ns + "/joint_states", JointState, timeout=0.1)
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                self.controllers_object.start_controllers(
                    controllers_on=f"joint_state_controller")
                rospy.logdebug(
                    "Current joint_states not ready yet, retrying==>"+str(e))

        for name in self.link_names:
            topic = '/' + self.ns + f'/{name}_collision_sensor'
            msg = None
            while msg is None and not rospy.is_shutdown():
                try:
                    msg = rospy.wait_for_message(
                        topic, ContactsState, timeout=0.1)
                    rospy.logdebug(f"Current {topic} READY")
                except Exception as e:
                    rospy.logdebug(
                        f"Current {topic} not ready yet, retrying==>"+str(e))
        rospy.logdebug('All collision sensors ready')

    def link_state_callback(self, msg):
        self.link_state = msg
        idx1 = msg.name.index('robot::robotiq_85_left_inner_knuckle_link')
        idx2 = msg.name.index('robot::robotiq_85_right_inner_knuckle_link')
        end_effector_pos = [(msg.pose[idx1].position.x + msg.pose[idx2].position.x) / 2, 
                            (msg.pose[idx1].position.y + msg.pose[idx2].position.y) / 2, 
                            (msg.pose[idx1].position.z + msg.pose[idx2].position.z) / 2]
        self._ur5_state.end_effector_position = end_effector_pos


    def joint_states_callback(self, msg):
        if msg.position and msg.velocity:
            self._ur5_state.joint_states = msg

    def contact_state_callback(self, msg):
        has_collision = False
        if msg.states:
            for state in msg.states:
                rospy.logdebug(f"Collision: {state.collision1_name}-{state.collision2_name}")
            has_collision = True
        if not self._ur5_state.had_collision:
            self._ur5_state.had_collision = has_collision

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def get_current_eef_position(self):
        """Get x,y,z coordinates according to currrent joint angles
        Returns:
        xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        rate = rospy.Rate(60)
        while self._ur5_state.end_effector_position is None:
            rate.sleep()
        return self._ur5_state.end_effector_position

    def get_current_joint_states(self):
        rate = rospy.Rate(60)
        while self._ur5_state.joint_states is None:
            rate.sleep()
        return self._ur5_state.joint_states

    def check_collisions(self):
        rate = rospy.Rate(60)
        while self._ur5_state.had_collision is None:
            rate.sleep()
        return self._ur5_state.had_collision

    def _generate_random_pose(self, limits):
        """ limits[0] - lower limits
            limits[1] - upper limits 
        """
        return np.random.uniform(limits[0], limits[1])
        