import numpy as np

from irs_rrt.irs_rrt import IrsNode
# from irs_rrt.irs_rrt_projection_3d import IrsRrtProjection3D
from irs_rrt.irs_rrt_projection_3d_rpy import IrsRrtProjection3DRPY
from irs_rrt.rrt_params import IrsRrtProjectionParams

from qsim_cpp import ForwardDynamicsMode
from qsim.parser import QuasistaticParser
from irs_rrt.contact_sampler_allegro import AllegroHandContactSampler
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer
from irs_mpc2.irs_mpc_params import SmoothingMode
from irs_rrt.allegro_3d_rpy_helper import convert_state_rpy_to_quat, convert_state_quat_to_rpy
from allegro_hand_setup import *

from pydrake.math import RollPitchYaw
from scipy.spatial.transform import Rotation as R


q_goal_path = '/home/jyp/research/inhand_manipulation/ros2_ws/src/inhand_lowlevel/leap_ros2/scripts/journal/data/quat_targets-250318.npy'
q_goal_data = np.load(q_goal_path, allow_pickle=True)
goal_idx = 16

# %%
q_parser = QuasistaticParser(q_model_path)
q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py
plant = q_sim.get_plant()

dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()
plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

contact_sampler = AllegroHandContactSampler(q_sim, q_sim_py)

q_a0 = np.array(
    # order: index, middle, ring, thumb
    [
        0.2, 0.95, 1.0, 1.0,
        0.0, 0.6, 1.0, 1.0,
        -0.2, 0.95, 1.0, 1.0,
        0.6, 1.95, 1.0, 1.0,
    ]
)

q_u0 = np.array([0.0, 0.0, 0.0])
# x0 = contact_sampler.sample_contact(q_u0)
q0 = q_sim.get_q_vec_from_dict({idx_u: q_u0, idx_a: q_a0})
q_vis.draw_configuration(q0)

contact_sampler.sample_contact(np.concatenate((q_u0, q_a0)))

num_joints = q_sim.num_actuated_dofs()
joint_limits = {
    # The first four elements correspond to quaternions. However, we are being
    # a little hacky and interpreting the first three elements as rpy here.
    # TODO(terry-suh): this doesn't seem to be consistent with the way joint
    # limits are defined for allegro_hand_traj_opt, since IrsRrtProjection3D
    # and IrsRrtTrajopt3D have different sample_subgoal functions. Should make
    # this slightly more consistent.
    idx_u: np.array(
        [
            [-np.pi, np.pi],        # roll
            [-np.pi/2, np.pi/2],    # pitch
            [-np.pi, np.pi],        # yaw
            [0, 0]
        ]
    ),
    idx_a: np.zeros([num_joints, 2]),
}

# %% IrsRrt params
rrt_params = IrsRrtProjectionParams(q_model_path, joint_limits)
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
rrt_params.root_node = IrsNode(convert_state_rpy_to_quat(q0))
rrt_params.max_size = 1000
rrt_params.goal = np.insert(np.copy(q0), 0, 0)
# rrt_params.goal[:4] = R.from_euler('xyz', [0, 0, np.pi/2]).as_quat()[[3, 0, 1, 2]]
rrt_params.goal[:4] = q_goal_data[goal_idx]
rrt_params.termination_tolerance = 0.01  # used in irs_rrt.iterate() as cost
# threshold.
rrt_params.goal_as_subgoal_prob = 0.3   # FIXME: was 0.3, 1.0 disables sampling subgoals
rrt_params.rewire = False
rrt_params.regularization = 1e-6
rrt_params.distance_metric = "quat_diff"    # FIXME: was "local_u"
# params.distance_metric = 'global'  # If using global metric
rrt_params.global_metric = np.ones(rrt_params.goal.shape) * 0.1
rrt_params.global_metric[:4] = [0, 0, 0, 0]
rrt_params.quat_metric = 5
rrt_params.distance_threshold = np.inf
rrt_params.stepsize = 0.2   # FIXME: was 0.2
rrt_params.std_u = 0.1
rrt_params.grasp_prob = 0.2     # FIXME: was 0.2, 0.0 disables regrasping
rrt_params.h = 0.1

# %% use free solvers?
use_free_solvers = True
rrt_params.use_free_solvers = use_free_solvers
contact_sampler.sim_params.use_free_solvers = use_free_solvers
# %% draw the goals
for i in range(1):
    prob_rrt = IrsRrtProjection3DRPY(rrt_params, contact_sampler, q_sim, q_sim_py)

    q_vis.draw_object_triad(
        length=0.1, radius=0.001, opacity=1, path="sphere/sphere"
    )

    prob_rrt.iterate()

    (
        q_knots_trimmed,
        u_knots_trimmed,
    ) = prob_rrt.get_trimmed_q_and_u_knots_to_goal()

    # quat --> rpy
    q_knots_rpy_trimmed = [convert_state_quat_to_rpy(q) for q in q_knots_trimmed]
    q_knots_rpy_trimmed = np.array(q_knots_rpy_trimmed)

    q_vis.publish_trajectory(q_knots_rpy_trimmed, h=rrt_params.h)

    # %%
    breakpoint()
    prob_rrt.save_tree(
        os.path.join(
            data_folder, "randomized", f"tree_{rrt_params.max_size}_{i}.pkl"
        )
    )
