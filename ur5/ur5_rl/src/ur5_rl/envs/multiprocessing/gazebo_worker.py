from __future__ import absolute_import

import multiprocessing as mp
import os
import atexit
import rospy

import roslaunch
from time import sleep

from ur5_rl.envs.task_envs.ur5_env_reach_goal import UR5EnvGoal


class GazeboRunner(mp.Process):
    def __init__(self, ros_master_port, gazebo_port, launch_files, **kwargs):
        super().__init__(**kwargs)
        self._ros_master_port = ros_master_port
        self._gazebo_port = gazebo_port
        self._launch_files = launch_files

    def run(self):
        os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{self._gazebo_port}'
        uuid = roslaunch.rlutil.get_or_generate_uuid(options_runid=None, options_wait_for_master=False)
        # roslaunch.configure_logging(uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_files=self._launch_files, is_core=False)
        atexit.register(self.launch.shutdown)
        self.launch.start(auto_terminate=False)  # start gazebo
        rospy.loginfo('Started gazebo runner')
        rospy.loginfo('Spinning')
        self.launch.spin()  # start event loop


class GazeboEnvWorker(mp.Process):
    def __init__(self, obs_slot, action_slot, ns, ros_master_port, gazebo_port, launch_files, **kwargs):
        super().__init__(daemon=kwargs.get('daemon', False))
        self._obs_slot = obs_slot
        self._action_slot = action_slot
        self._ros_master_port = ros_master_port
        self._gazebo_port = gazebo_port
        self._ns = ns
        self._launch_files = [(file, f'ns:={self._ns}') for file in launch_files]
        self._controllers_list = kwargs.get('controllers_list', [])
        self._link_names = kwargs.get('link_names', [])
        self._joint_limits = kwargs.get('joint_limits', [])
        self._target_limits = kwargs.get('target_limits', [])
        self._pub_topic_name = kwargs.get('pub_topic_name', [])

        self.__env = None
        self.__done = True

    def run(self):
        os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{self._gazebo_port}'
        self.runner = GazeboRunner(self._ros_master_port, self._gazebo_port, self._launch_files, daemon=True)
        atexit.register(self.runner.terminate)
        self.runner.start()  # start gazebo event loop
        sleep(5)
        rospy.loginfo('Finished starting gazebo')
        rospy.init_node('worker', anonymous=True, log_level=rospy.DEBUG)
        self.__env = UR5EnvGoal(self._ns,
                                self._controllers_list, self._link_names,
                                self._joint_limits, self._target_limits,
                                self._pub_topic_name)
        rospy.loginfo('Initialized everything')
        while self.runner.is_alive():  # process messages
            action = self._action_slot.get().clone()
            if self.__done or action == 'reset':
                self.__done = False
                self._obs_slot.put([self.__env.reset(), 0.0, False, {}])
            else:
                obs, reward, done, info = self.__env.step(action)
                self.__done = done
                self._obs_slot.put([obs, reward, done, info])
