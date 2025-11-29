import argparse
import copy
from typing import List
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (
    RigidTransform,
    Quaternion,
    RollPitchYaw,
    AngleAxis,
    PiecewisePolynomial,
)

from qsim_cpp import ForwardDynamicsMode, GradientMode
from qsim.parser import QuasistaticParser

import irs_rrt
# from irs_rrt.irs_rrt import IrsRrt
from irs_rrt.irs_rrt_projection_3d_rpy import IrsRrtProjection3DRPY
from irs_mpc2.quasistatic_visualizer import (
    QuasistaticVisualizer,
    InternalVisualizationType,
)
from irs_rrt.allegro_3d_rpy_helper import (
    convert_state_quat_to_rpy,
    convert_state_rpy_to_quat
)

# from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_3d_rpy import IrsMpc3DRPYQuasistatic
from irs_mpc2.irs_mpc_params import SmoothingMode, IrsMpcQuasistaticParameters
from allegro_hand_setup import robot_name, object_name, q_model_path


MAX_TRAJOPT_ATTEMPTS = 10

def solve_allegro_sphere_rpy_trajopt(goal_idx=0, interact=True, rand_seed=42):
    np.random.seed(rand_seed)

    # %%
    pickled_tree_path = os.path.join(
        os.path.dirname(irs_rrt.__file__),
        "..",
        "ptc_data",
        "allegro_hand_sphere_rpy",
        "rrt_planned",
        # "tree_1000_0.pkl",
        f"goal_{goal_idx}.pkl",
    )

    saved_result_path = os.path.join(
        os.path.dirname(irs_rrt.__file__),
        "..",
        "ptc_data",
        "allegro_hand_sphere_rpy",
        "rrt_trajopt_refined",
        f"goal_{goal_idx}.pkl",
    )

    if not os.path.exists(pickled_tree_path):
        raise FileNotFoundError(f"Pickled tree not found at {pickled_tree_path}")

    with open(pickled_tree_path, "rb") as f:
        tree = pickle.load(f)

    prob_rrt = IrsRrtProjection3DRPY.make_from_pickled_tree(
        tree, internal_vis=InternalVisualizationType.Cpp
    )

    q_sim, q_sim_py = prob_rrt.q_sim, prob_rrt.q_sim_py
    q_vis = QuasistaticVisualizer(q_sim=q_sim, q_sim_py=q_sim_py)

    # get goal and some problem data from RRT parameters.
    q_u_goal = prob_rrt.rrt_params.goal[:4]
    Q_WB_d = Quaternion(q_u_goal)
    p_WB_d = np.array([-0.06, 0.0, 0.072])
    dim_q = prob_rrt.dim_q
    dim_u = q_sim.num_actuated_dofs()

    # visualize goal.
    q_vis.draw_object_triad(
        length=0.1,
        radius=0.001,
        opacity=1,
        path="sphere/sphere",
    )
    q_vis.draw_goal_triad(
        length=0.1, radius=0.005, opacity=0.7, X_WG=RigidTransform(Q_WB_d, p_WB_d)
    )

    # %%
    # get trimmed path to goal.
    q_knots_trimmed, u_knots_trimmed = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
    # split trajectory into segments according to re-grasps.
    segments = prob_rrt.get_regrasp_segments(u_knots_trimmed)

    q_knots_rpy_trimmed = [convert_state_quat_to_rpy(q) for q in q_knots_trimmed]
    q_knots_rpy_trimmed = np.array(q_knots_rpy_trimmed)

    # %% see the segments.
    prob_rrt.print_segments_displacements(q_knots_trimmed, segments)
    q_vis.publish_trajectory(q_knots_rpy_trimmed, prob_rrt.rrt_params.h)

    # %% determining h_small and n_steps_per_h from simulating a segment.
    h_small = 0.01

    # %% IrsMpc
    q_parser = QuasistaticParser(prob_rrt.rrt_params.q_model_path)
    plant = q_sim.get_plant()
    indices_q_u_into_x = [0, 1, 2, 3]
    indices_q_a_into_x = list(range(4, 4+dim_u))
    idx_a = plant.GetModelInstanceByName(robot_name)
    idx_u = plant.GetModelInstanceByName(object_name)

    # traj-opt parameters
    impc_params = IrsMpcQuasistaticParameters()
    impc_params.enforce_joint_limits = (
        prob_rrt.rrt_params.enforce_robot_joint_limits
    )

    impc_params.h = h_small
    impc_params.Q_dict = {
        idx_u: np.array([10, 10, 10, 10]),  # quaternion
        idx_a: np.ones(dim_u) * 1e-3,
    }

    impc_params.Qd_dict = {}
    for model in q_sim.get_actuated_models():
        impc_params.Qd_dict[model] = impc_params.Q_dict[model]
    for model in q_sim.get_unactuated_models():
        impc_params.Qd_dict[model] = impc_params.Q_dict[model] * 200

    impc_params.R_dict = {idx_a: 10 * np.ones(dim_u)}

    u_size = 5.0
    impc_params.u_bounds_abs = np.array(
        [
            -np.ones(dim_u) * u_size * impc_params.h,
            np.ones(dim_u) * u_size * impc_params.h,
        ]
    )

    impc_params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
    # sampling-based bundling
    impc_params.calc_std_u = lambda u_initial, i: u_initial / (i**0.8)
    impc_params.std_u_initial = np.ones(dim_u) * 0.3
    impc_params.num_samples = 100
    # analytic bundling
    impc_params.log_barrier_weight_initial = 100
    log_barrier_weight_final = 6000
    max_iterations = 15

    base = (
        np.log(log_barrier_weight_final / impc_params.log_barrier_weight_initial)
        / max_iterations
    )
    base = np.exp(base)
    impc_params.calc_log_barrier_weight = lambda kappa0, i: kappa0 * (base**i)

    impc_params.use_A = False
    impc_params.rollout_forward_dynamics_mode = ForwardDynamicsMode.kSocpMp
    prob_mpc = IrsMpc3DRPYQuasistatic(q_sim=q_sim, parser=q_parser, params=impc_params)

    # %% traj-opt for segment
    sim_params_projection = copy.deepcopy(prob_rrt.sim_params)
    sim_params_projection.unactuated_mass_scale = 1e-4


    def project_to_non_penetration(q: np.ndarray):
        return q_sim.calc_dynamics(q, q[-16:], sim_params_projection)


    q_trj_optimized_list = []
    u_trj_optimized_list = []

    sub_segments = segments[0:]

    seg_trajopt_succ_flag = []

    if interact:
        input("Starting refinement...")
    else:
        print("Starting refinement...")

    for i_s, (t_start, t_end) in enumerate(sub_segments):
        u_trj = u_knots_trimmed[t_start:t_end]
        q_trj = q_knots_trimmed[t_start : t_end + 1]

        q_traj_rpy = [convert_state_quat_to_rpy(q) for q in q_trj]
        q_vis.publish_trajectory(q_traj_rpy, prob_rrt.rrt_params.h)

        q0 = np.array(q_trj[0])
        if len(q_trj_optimized_list) > 0:
            q0[indices_q_u_into_x] = q_trj_optimized_list[-1][
                -1, indices_q_u_into_x
            ]
            print("qu0 before projection", q0[indices_q_u_into_x])
            q0 = project_to_non_penetration(convert_state_quat_to_rpy(q0))
            q0 = convert_state_rpy_to_quat(q0)
            print("qu0 after projection", q0[indices_q_u_into_x])

        if interact:
            input("Original trajectory segment shown. Press any key to optimize...")
        else:
            print("Original trajectory segment shown. Optimizing...")

        q_final = np.array(q_trj[-1])
        if i_s == len(sub_segments) - 1:
            q_final[indices_q_u_into_x] = q_u_goal

        n_steps_per_h = max(2, int(np.ceil(10 / len(u_trj))))

        num_attempts = 0
        flag_trajopt_succ = False
        while num_attempts < MAX_TRAJOPT_ATTEMPTS:
            try:
                (
                    q_trj_optimized,
                    u_trj_optimized,
                    idx_best,
                ) = prob_mpc.run_traj_opt_on_rrt_segment(
                    n_steps_per_h=n_steps_per_h,
                    h_small=h_small,
                    q0=q0,
                    q_final=q_final,
                    u_trj=u_trj,
                    max_iterations=max_iterations,
                )
                flag_trajopt_succ = True
                break
            except:
                print("Trajectory optimization failed, retrying...")
            num_attempts += 1

        if not flag_trajopt_succ:
            print("Trajectory optimization failed... using original RRT trajectory")
            q_trj_optimized = q_trj
            u_trj_optimized = u_trj

        seg_trajopt_succ_flag.append(flag_trajopt_succ)

        q_trj_optimized_list.append(q_trj_optimized)
        u_trj_optimized_list.append(u_trj_optimized)

        q_trj_optimized_rpy = [convert_state_quat_to_rpy(q) for q in q_trj_optimized]

        if interact:
            prob_mpc.plot_costs()

        q_vis.publish_trajectory(q_trj_optimized_rpy, h_small)
        print(f"Best trajectory iteration index: {idx_best}")

        if interact:
            input("Optimized trajectory shown. Press any key to go to the next segment")
        else:
            print("Optimized trajectory shown. Going to the next segment")

    # %%
    q_trj_optimized_all = prob_rrt.concatenate_traj_list(q_trj_optimized_list)

    q_trj_optimized_all_rpy = [convert_state_quat_to_rpy(q) for q in q_trj_optimized_all]
    q_vis.publish_trajectory(q_trj_optimized_all_rpy, 0.1)

    # # %% see differences between RRT and optimized trajectories.
    # for t, q_trj_optimized in enumerate(q_trj_optimized_list):
    #     # prob_mpc.q_vis.publish_trajectory(q_trj_optimized, h_small)

    #     t_end = segments[t][1]
    #     q_u_d = q_knots_trimmed[t_end, q_sim.get_q_u_indices_into_q()]
    #     q_u_final = q_trj_optimized[-1][indices_q_u_into_x]
    #     angle_diff, position_diff = prob_rrt.calc_q_u_diff(q_u_final, q_u_d)
    #     print("angle diff", angle_diff, "position diff", position_diff)

    # %%
    print("Trimming optimized trajectory segments")
    q_trj_optimized_trimmed_list = []
    u_trj_optimized_trimmed_list = []
    for q_trj, u_trj in zip(q_trj_optimized_list, u_trj_optimized_list):
        t = prob_rrt.trim_trajectory(q_trj)
        print(f"{t} / {len(q_trj)}")
        q_trj_optimized_trimmed_list.append(q_trj[: t + 1])
        u_trj_optimized_trimmed_list.append(u_trj[:t])

    q_trj_optimized_trimmed_all = prob_rrt.concatenate_traj_list(
        q_trj_optimized_trimmed_list
    )
    q_trj_optimized_trimmed_all_rpy = [convert_state_quat_to_rpy(q) for q in q_trj_optimized_trimmed_all]
    q_vis.publish_trajectory(q_trj_optimized_trimmed_all_rpy, 0.1)

    # %%
    with open(saved_result_path, "wb") as f:
        pickle.dump(
            {
                "q_trj_list": q_trj_optimized_trimmed_list,
                "u_trj_list": u_trj_optimized_trimmed_list,
                "h_small": h_small,
            },
            f,
        )

    trajopt_succ = np.sum(seg_trajopt_succ_flag) == len(seg_trajopt_succ_flag)

    return trajopt_succ

def visualize_trajopt_solution(goal_idx=0):
    def concatenate_traj_list(q_trj_list: List[np.ndarray]):
        """
        Concatenates a list of trajectories into a single trajectory.
        q_trj_list[i] has shape (T_i, n_q)
        """
        q_trj_sizes = np.array([len(q_trj) for q_trj in q_trj_list])
        dim_q = q_trj_list[0].shape[1]
        q_trj_all = np.zeros((q_trj_sizes.sum(), dim_q))

        t_start = 0
        for q_trj_size, q_trj in zip(q_trj_sizes, q_trj_list):
            q_trj_all[t_start : t_start + q_trj_size] = q_trj
            t_start += q_trj_size

        return q_trj_all
    
    saved_result_path = os.path.join(
        os.path.dirname(irs_rrt.__file__),
        "..",
        "ptc_data",
        "allegro_hand_sphere_rpy",
        "rrt_trajopt_refined",
        f"goal_{goal_idx}.pkl",
    )

    data = pickle.load(open(saved_result_path, "rb"))

    goal_path = '/home/jyp/research/inhand_manipulation/ros2_ws/src/inhand_lowlevel/leap_ros2/scripts/journal/data/quat_targets-250318.npy'
    Q_WB_d = Quaternion(np.load(goal_path, allow_pickle=True)[goal_idx])
    p_WB_d = np.array([-0.06, 0.0, 0.072])

    q_parser = QuasistaticParser(q_model_path)
    q_vis = QuasistaticVisualizer.make_visualizer(q_parser)

    q_vis.draw_goal_triad(
        length=0.1, radius=0.005, opacity=0.7, X_WG=RigidTransform(Q_WB_d, p_WB_d)
    )

    q_trj_optimized_trimmed_all = concatenate_traj_list(data["q_trj_list"])
    q_trj_optimized_trimmed_all_rpy = [convert_state_quat_to_rpy(q) for q in q_trj_optimized_trimmed_all]
    q_vis.publish_trajectory(q_trj_optimized_trimmed_all_rpy, 0.1)
    breakpoint()



if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal_idx", type=int, default=0, help="Index of goal to plan to")
    parser.add_argument("--rand_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--action", type=str, default="solve", help="solve or visualize")
    args = parser.parse_args()

    if args.action == "solve":
        succ = solve_allegro_sphere_rpy_trajopt(goal_idx=args.goal_idx, interact=False, rand_seed=args.rand_seed)
    elif args.action == "visualize":
        visualize_trajopt_solution(goal_idx=args.goal_idx)
        succ = True

    if succ:
        exit(0)
    else:
        exit(1)
