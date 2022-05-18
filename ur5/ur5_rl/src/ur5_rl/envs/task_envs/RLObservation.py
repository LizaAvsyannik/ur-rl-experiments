import torch

class RLObservation:
    def __init__(self, joint_states, end_effector_position, joint_limits, target_position):
        self._lower_limits = torch.Tensor(joint_limits['lower'])
        self._upper_limits = torch.Tensor(joint_limits['upper'])
        pis = torch.pi * torch.ones(6)
        self._joint_positions = torch.fmod(torch.Tensor(joint_states.position[:6]),
                                           4 * pis) - 2 * pis
        self._joint_velocities = torch.Tensor(joint_states.velocity[:6])
        self._end_effector_pos = torch.Tensor(end_effector_position)
        self._target_position = torch.Tensor(target_position)

    def get_model_input(self):
        return torch.hstack([self._joint_positions, self._joint_velocities,
                            #  self._joint_positions - self._lower_limits,
                            #  self._upper_limits - self._joint_positions,
                             self._end_effector_pos, self._target_position])

    @staticmethod
    def dim():
        return 18
