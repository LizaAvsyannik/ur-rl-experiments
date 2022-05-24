from __future__ import absolute_import

import multiprocessing as mp
import os
import atexit
import rospy

import roslaunch
from time import sleep


class GazeboRunner(mp.Process):
    def __init__(self, gazebo_port, launch_files, **kwargs):
        super().__init__(**kwargs)
        self._gazebo_port = gazebo_port
        self._launch_files = launch_files

    def run(self):
        os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{self._gazebo_port}'
        uuid = roslaunch.rlutil.get_or_generate_uuid(options_runid=None, options_wait_for_master=False)
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_files=self._launch_files, is_core=False)
        atexit.register(self.launch.shutdown)
        self.launch.start(auto_terminate=False)  # start gazebo
        rospy.loginfo('Started gazebo runner')
        rospy.loginfo('Spinning')
        self.launch.spin_once()  # start event loop


class GazeboEnvWorker(mp.Process):
    def __init__(self, env_cls, obs_slot, action_slot, ns, gazebo_port, launch_files, **kwargs):
        super().__init__(daemon=kwargs.get('daemon', False))
        self._env_cls = env_cls
        self._obs_slot = obs_slot
        self._action_slot = action_slot
        self._gazebo_port = gazebo_port
        self._ns = ns
        self._launch_files = [(file, [f'namespace:={self._ns}']) for file in launch_files]
        self._env_kwargs = kwargs
        self._env_kwargs['ns'] = self._ns

        self.__env = None
        self.__done = True

    def run(self):
        os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{self._gazebo_port}'
        self.runner = GazeboRunner(self._gazebo_port, self._launch_files, daemon=True)
        atexit.register(self.runner.terminate)
        self.runner.start()  # start gazebo event loop
        sleep(5)
        rospy.init_node('worker', anonymous=True, log_level=rospy.WARN)
        rospy.loginfo('Finished starting gazebo')
        self.__env = self._env_cls(**self._env_kwargs)
        rospy.loginfo('Initialized everything')
        while True:  # process messages
            action = self._action_slot.get()
            if self.__done or action == 'reset':
                self.__done = False
                rospy.logwarn(f'Resetting in {self._ns}')
                self._obs_slot.put([self.__env.reset(), 0.0, False, {}])
            else:
                rospy.logdebug('Stepping')
                obs, reward, done, info = self.__env.step(action.clone())
                self.__done = done
                self._obs_slot.put([obs, reward, done, info])
