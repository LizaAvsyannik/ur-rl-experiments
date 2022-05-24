import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties, GetModelState
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3


class GazeboConnection():
    
    def __init__(self, ns):
        rospy.logdebug('Initializaing Gazebo Connection...')
        self.ns = ns
        self._get_physics = rospy.ServiceProxy('/' + self.ns + '/gazebo/get_physics_properties', GetPhysicsProperties)
        self._set_physics = rospy.ServiceProxy('/' + self.ns + '/gazebo/set_physics_properties', SetPhysicsProperties)
        
        rospy.wait_for_service('/' + self.ns + '/gazebo/get_physics_properties')
        self._physics_properties = self._get_physics()

        self._unpause = rospy.ServiceProxy('/' + self.ns + '/gazebo/unpause_physics', Empty)
        self._pause = rospy.ServiceProxy('/' + self.ns + '/gazebo/pause_physics', Empty)
        self._reset_simulation_proxy = rospy.ServiceProxy('/' + self.ns + '/gazebo/reset_simulation', Empty)
        self._reset_world_proxy = rospy.ServiceProxy('/' + self.ns + '/gazebo/reset_world', Empty)

        self._get_model_state = rospy.ServiceProxy(
            '/' + self.ns + "/gazebo/get_model_state", GetModelState)

        # We always pause the simulation, important for legged robots learning
        self.pauseSim()
        rospy.logdebug('Initialized Gazebo Connection')

    def pauseSim(self):
        rospy.logdebug("PAUSING START")
        rospy.wait_for_service('/' + self.ns + '/gazebo/pause_physics')
        try:
            self._pause()
        except rospy.ServiceException as e:
            rospy.logerr('/' + self.ns + "/gazebo/pause_physics service call failed")
            
        rospy.logdebug("PAUSING FINISH")
        
    def unpauseSim(self):
        rospy.logdebug("UNPAUSING START")
        rospy.wait_for_service('/' + self.ns + '/gazebo/unpause_physics')
        try:
            self._unpause()
        except rospy.ServiceException as e:
            rospy.logerr('/' + self.ns + "/gazebo/unpause_physics service call failed")
        
        rospy.logdebug("UNPAUSING FiNISH")
    
    def resetSimulation(self):
        rospy.wait_for_service('/' + self.ns + '/gazebo/reset_simulation')
        try:
            self._reset_simulation_proxy()
        except rospy.ServiceException as e:
            rospy.logerr('/' + self.ns + "/gazebo/reset_simulation service call failed")

    def resetWorld(self):
        rospy.wait_for_service('/' + self.ns + '/gazebo/reset_world')
        try:
            self._reset_world_proxy()
        except rospy.ServiceException as e:
            rospy.logerr('/' + self.ns + "/gazebo/reset_world service call failed")

    def set_gravity_to_default(self):
        rospy.wait_for_service('/' + self.ns + '/gazebo/get_physics_properties')
        self._physics_properties = self._get_physics()
        
        rospy.wait_for_service('/' + self.ns + '/gazebo/set_physics_properties')
        try:
            gravity = Vector3()
            gravity.x = 0.0
            gravity.y = 0.0
            gravity.z = -9.81
            response = self._set_physics(self._physics_properties.time_step,
                                         self._physics_properties.pause,
                                         gravity,
                                         self._physics_properties.ode_config)
        except rospy.ServiceException:
            rospy.logerr('/' + self.ns + "/gazebo/set_physics_properties service call failed")

    def disable_gravity(self):
        rospy.wait_for_service('/' + self.ns + '/gazebo/get_physics_properties')
        self._physics_properties = self._get_physics()
        
        rospy.wait_for_service('/' + self.ns + '/gazebo/set_physics_properties')
        try:
            gravity = Vector3()
            gravity.x = 0.0
            gravity.y = 0.0
            gravity.z = 0.0
            response = self._set_physics(self._physics_properties.time_step,
                                         self._physics_properties.pause,
                                         gravity,
                                         self._physics_properties.ode_config)
        except rospy.ServiceException:
            rospy.logerr('/' + self.ns + "/change_gravity_zero service call failed")

    def get_model_state(self, args):
        rospy.wait_for_service('/' + self.ns + '/gazebo/get_model_state')
        try:
            return self._get_model_state(*args)
        except rospy.ServiceException as e:
            rospy.logerr('/' + self.ns + "/gazebo/get_model_state service call failed")
            