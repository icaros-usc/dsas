# Goals for prediction and assistance
# pose corresponds to the pose of the goal object
# target poses corresponds to all the grasp locations for this object

import numpy as np
from tf import transformations as transmethods

from ada_assistance_policy.Utils import QuaternionDistance


class Goal:
    def __init__(self, pose, target_poses=list(), target_iks=list()):
        self.pose = pose
        self.pos = pose[0:3, 3]

        if not target_poses:
            target_poses.append(pose)

        # copy the targets
        self.target_poses = list(target_poses)
        self.target_iks = list(target_iks)

        self.target_quaternions = [
            transmethods.quaternion_from_matrix(target_pose)
            for target_pose in self.target_poses
        ]

    def at_goal(
        self, end_effector_trans, trans_threshold=0.01, quat_threshold=np.pi / 48
    ):
        for pose, quat in zip(self.target_poses, self.target_quaternions):
            pos_diff = pose[0:3, 3] - end_effector_trans[0:3, 3]
            trans_dist = np.linalg.norm(pos_diff)
            quat_dist = QuaternionDistance(
                transmethods.quaternion_from_matrix(end_effector_trans), quat
            )

            if (trans_dist < trans_threshold) and (quat_dist < quat_threshold):
                return True

        # if none of the poses in target_poses returned, then we are not at goal
        return False
