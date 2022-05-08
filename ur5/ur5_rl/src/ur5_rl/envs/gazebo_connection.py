import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3


class GazeboConnection():
    
    def __init__(self):
        rospy.logdebug('Initializaing Gazebo Connection...')
        self._get_physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        self._set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        
        rospy.wait_for_service('/gazebo/get_physics_properties')
        self._physics_properties = self._get_physics()

        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self._reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # We always pause the simulation, important for legged robots learning
        self.pauseSim()
        rospy.logdebug('Initialized Gazebo Connection')

    def pauseSim(self):
        rospy.logdebug("PAUSING START")
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self._pause()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/pause_physics service call failed")
            
        rospy.logdebug("PAUSING FINISH")
        
    def unpauseSim(self):
        rospy.logdebug("UNPAUSING START")
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self._unpause()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/unpause_physics service call failed")
        
        rospy.logdebug("UNPAUSING FiNISH")
    
    def resetSimulation(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self._reset_simulation_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_simulation service call failed")

    def resetWorld(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self._reset_world_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_world service call failed")

    def set_gravity_to_default(self):
        rospy.wait_for_service('/gazebo/get_physics_properties')
        self._physics_properties = self._get_physics()
        
        rospy.wait_for_service('/gazebo/set_physics_properties')
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
            rospy.logerr("/gazebo/set_physics_properties service call failed")

    def disable_gravity(self):
        rospy.wait_for_service('/gazebo/get_physics_properties')
        self._physics_properties = self._get_physics()
        
        rospy.wait_for_service('/gazebo/set_physics_properties')
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
            rospy.logerr("/change_gravity_zero service call failed")