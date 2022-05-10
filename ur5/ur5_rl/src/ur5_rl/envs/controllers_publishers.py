import rospy
from std_msgs.msg import Float64MultiArray


class Pub(object):
    def __init__(self, topic_name, cntrl_conn):
         self.topic_name = topic_name # /<cntrl_name>/command
         self.cntrl_conn = cntrl_conn

    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        raise NotImplementedError
    
    def move_joints(self, joints_array):
        raise NotImplementedError

    
class JointGroupPublisher(Pub):
    def __init__(self, topic_name, cntrl_conn):
        """
        Publish std_msgs::Float64MultiArray for position/velocity group control
        """
        Pub.__init__(self, topic_name=topic_name, cntrl_conn=cntrl_conn)
        rospy.logdebug('Initializaing Joint Publishers...')

        self._pub = rospy.Publisher(
            self.topic_name, Float64MultiArray, queue_size=1)
        
        self.check_publishers_connection()

        rospy.logdebug('Initialized Joint Publishers...')


    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(1)  # 1hz
        while (self._pub.get_num_connections() == 0):
            rospy.logdebug(
                "No subscribers to joint_group_position_controller yet so we wait and try again")
            try:
                self.cntrl_conn.start_controllers(
                    controllers_on=self.topic_name[1:-8])
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("Publisher Connected")

        rospy.logdebug("All Joint Publishers READY")

    def move_joints(self, joints_array):
        pose = Float64MultiArray()

        rate = rospy.Rate(10)
        while self._pub.get_num_connections() < 1:
            rospy.logdebug("Waiting for connection")
        pose.data = list(joints_array)
        rate.sleep()

        self._pub.publish(pose)
        rospy.logdebug('Moved joints')




