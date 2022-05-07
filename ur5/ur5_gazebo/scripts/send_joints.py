#!/usr/bin/env python3
#
# Send joint values to UR5 using messages
#

from std_msgs.msg import Float64MultiArray
import rospy


def main():

    rospy.init_node('send_joints')
    pub = rospy.Publisher('/joint_group_position_controller/command',
                          Float64MultiArray, queue_size=1)

    # Create the topic message
    pose = Float64MultiArray()

    while pub.get_num_connections() < 1:
        rospy.logdebug("Waiting for connection")
    pose.data = [0.0, 0.0, 0.0, 0.0, 0.0, -0.2]

    pub.publish(pose)
    print(pose)



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")
