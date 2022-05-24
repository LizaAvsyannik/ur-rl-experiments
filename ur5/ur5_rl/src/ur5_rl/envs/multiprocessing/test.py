import yaml
import rospy
import torch

from ur5_rl.envs.multiprocessing.gazebo_master import GazeboMaster

if __name__ == '__main__':
    NENVS = 1
    ros_master_ports = [10350 + i for i in range(NENVS)]
    gazebo_ports = [10450 + i for i in range(NENVS)]
    launch_files = ['/home/ros/catkin_ws/src/ur-rl-experiments/ur5/ur5_gazebo/launch/ur5_cubes.launch']
    # rospy.init_node('test', log_level=rospy.DEBUG)

    params_path = '/home/ros/catkin_ws/src/ur-rl-experiments/ur5/ur5_rl/config/reach_goal_params.yaml'
    with open(params_path, "r") as params_stream:
        env_kwargs = yaml.safe_load(params_stream)
        env_kwargs['pub_topic_name'] = f'/{env_kwargs["controllers_list"][0]}/command'
        master = GazeboMaster(NENVS, ros_master_ports, gazebo_ports, launch_files, **env_kwargs)
        master.start()
        rospy.loginfo('Started master')
        master.reset()
        for i in range(10):
            print(master.get_observations())
            master.step(torch.ones((NENVS, 6)))
