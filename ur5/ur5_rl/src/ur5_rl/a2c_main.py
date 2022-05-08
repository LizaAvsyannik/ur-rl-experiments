 #!/usr/bin/env python3

import rospy
from ur5_rl.envs.task_envs import UR5EnvGoal

rospy.init_node('ur5_gym', anonymous=False, log_level=rospy.DEBUG)

controllers_list = ['joint_state_controller', 'joint_group_position_controller',  'gripper_controller']

joint_limits = [[-3.14]*6, [3.14]*6]
target_limits = [[0.275, -0.435, 0.775], [0.7, 0.435,  0.775], [0.05, 0.05, 0.05]]
env = UR5EnvGoal(controllers_list, joint_limits, target_limits, 'joint_group_position_controller/command')


print(env.reset())