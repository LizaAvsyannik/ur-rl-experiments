import rospy
from ur5_rl.envs import RobotGazeboEnv
from ur5_rl.envs import JointGroupPublisher
from gazebo_msgs.msg import LinkStates, ModelState
from sensor_msgs.msg import JointState
from ur5_rl.envs.ur_setups import setups
from ur5_rl.envs import ur_utils


class UR5Env(RobotGazeboEnv):
    def __init__(self, controllers_list, joint_limits, target_limits):
        rospy.logdebug("Start UR5Env INIT...")
        
        self.controllers_list = controllers_list
        self.joint_limits = joint_limits
        self.target_limits = target_limits #0 row - lower limits, 1 row - upper, 2nd - cube size

         # Arm/Control parameters
        self._ik_params = setups['UR5_6dof']['ik_params']

        RobotGazeboEnv.__init__(self, controllers_list=self.controllers_list)


        rospy.logdebug("UR5Env unpause...")
        self.gazebo.unpauseSim()

        # Subscribe link and joint states 
        self._get_link_states = rospy.Subscriber("/gazebo/link_states", LinkStates,
                            self.link_state_callback, queue_size=1)

        self._get_joint_states = rospy.Subscriber("/joint_states", JointState,
                            self.joints_state_callback, queue_size=1)

        self._set_model_state = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)

        rospy.logdebug("Subscribed to states")

        self.controllers_object.reset_controllers(self.controllers_list)

        self._check_all_systems_ready()
        
        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished UR5Env INIT...")

    def _check_all_systems_ready(self):
        """
        We check that all systems are ready
        :return:
        """
        link_states_msg = None
        while link_states_msg is None and not rospy.is_shutdown():
            try:
                link_states_msg = rospy.wait_for_message(
                    "/gazebo/link_states", LinkStates, timeout=0.1)
                self.link_state = link_states_msg
                rospy.logdebug("Current link_states READY")
            except Exception as e:
                rospy.logdebug(
                    "Current links not ready yet, retrying==>"+str(e))

        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message(
                    "/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                # self._ctrl_conn.load_controllers("joint_state_controller")
                self.controllers_object.start_controllers(
                    controllers_on=f"joint_state_controller")
                rospy.logdebug(
                    "Current joint_states not ready yet, retrying==>"+str(e))

        while (self._set_model_state.get_num_connections() == 0):
            rospy.logdebug(
                "No publishers to /gazebo/set_link_state yet so we wait and try again")

        rospy.logdebug("ALL SYSTEMS READY")

    def link_state_callback(self, msg):
        self.link_state = msg
        self.end_effector = self.link_state.pose[8]

    def joints_state_callback(self, msg):
        self.joints_state = msg

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

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def get_xyz(self, q, ik_params):
        """Get x,y,z coordinates 
        Args:
            q: a numpy array of joints angle positions.
        Returns:
            xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        mat = ur_utils.forward(q, ik_params)
        xyz = mat[:3, 3]
        return xyz

    def get_current_xyz(self):
        """Get x,y,z coordinates according to currrent joint angles
        Returns:
        xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message(
                    "/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
            except Exception as e:
                self.controllers_object.start_controllers(
                    controllers_on="joint_state_controller")
                rospy.logdebug(
                    "Current joint_states for get_current_xyz not ready yet, retrying==>"+str(e))
        self.joints_state = joint_states_msg
        end_effector_xyz = self.get_xyz(self.joints_state.position[:6], self._ik_params)
        return end_effector_xyz