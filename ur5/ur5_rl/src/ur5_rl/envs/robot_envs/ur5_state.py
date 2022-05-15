class UR5State:
    def __init__(self):
        self._joint_states = None
        self._end_effector_position = None
        self._had_collision = None

    def reset(self):
        self._joint_states = None
        self._end_effector_position = None
        self._had_collision = None

    @property
    def joint_states(self):
        return self._joint_states

    @joint_states.setter
    def joint_states(self, joint_states):
        self._joint_states = joint_states

    @property
    def end_effector_position(self):
        return self._end_effector_position

    @end_effector_position.setter
    def end_effector_position(self, position):
        self._end_effector_position = position

    def update(self, joint_states, eef_position):
        self._joint_states = joint_states
        self._end_effector_position = eef_position

    @property
    def had_collision(self):
        return self._had_collision

    @had_collision.setter
    def had_collision(self, had_collision):
        self._had_collision = had_collision
