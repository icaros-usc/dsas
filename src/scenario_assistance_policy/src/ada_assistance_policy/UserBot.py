import numpy as np


class UserBot:
    def __init__(self):
        # scaling factors for generating commands
        self.position_scale_vector = 10
        self.clip_norm_val = 0.25

    def get_usr_cmd(self, end_effector_trans, goal_pose):
        pos_diff = self.position_scale_vector * (
            goal_pose[0:3, 3] - end_effector_trans[0:3, 3]
        )

        pos_diff[2] = 0
        pos_diff_norm = np.linalg.norm(pos_diff)

        if pos_diff_norm > self.clip_norm_val:
            pos_diff /= pos_diff_norm / self.clip_norm_val

        return pos_diff
