import numpy as np
from tqdm import tqdm

from pydrake.all import RollPitchYaw, Quaternion, RotationMatrix

from qsim.simulator import QuasistaticSimulator
from qsim_cpp import QuasistaticSimulatorCpp

from irs_rrt.rrt_params import IrsRrtProjectionParams
from irs_rrt.irs_rrt_3d_rpy import IrsRrt3DRPY
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.contact_sampler import ContactSampler
from irs_rrt.irs_rrt import IrsRrtParams, IrsRrt, IrsNode, IrsEdge
from irs_rrt.rrt_base import Node
from irs_rrt.allegro_3d_rpy_helper import (
    convert_state_quat_to_rpy,
    convert_state_rpy_to_quat,
    quat_angle_difference
)

from scipy.spatial.transform import Rotation as R, Slerp


class IrsRrtProjection3DRPY(IrsRrtProjection):
    def __init__(
        self,
        rrt_params: IrsRrtProjectionParams,
        contact_sampler: ContactSampler,
        q_sim: QuasistaticSimulatorCpp,
        q_sim_py: QuasistaticSimulator,
    ):
        super().__init__(rrt_params, contact_sampler, q_sim, q_sim_py, use_rpy_reachable_set=True)
        self.irs_rrt_3d = IrsRrt3DRPY(rrt_params, self.q_sim, q_sim_py)

    def sample_random_subgoal(self):
        # Sample translation
        rand_num = np.random.rand(16)
        subgoal = np.zeros(20)
        subgoal[-16:] = self.q_lb[-16:] + (self.q_ub[-16:] - self.q_lb[-16:]) * rand_num

        quat = R.random().as_quat()[[3, 0, 1, 2]]
        subgoal[self.irs_rrt_3d.quat_ind] = quat

        return subgoal
    
    def sample_subgoal_between_start_goal(self):
        start_q = self.rrt_params.root_node.q
        goal_q = self.rrt_params.goal

        rand_num = np.random.rand(16)
        subgoal = np.zeros(20)
        subgoal[-16:] = self.q_lb[-16:] + (self.q_ub[-16:] - self.q_lb[-16:]) * rand_num

        quat_start = R.from_quat(start_q[self.irs_rrt_3d.quat_ind][[1, 2, 3, 0]])
        quat_goal = R.from_quat(goal_q[self.irs_rrt_3d.quat_ind][[1, 2, 3, 0]])
        slerp_frac = np.random.rand()

        slerp_obj = Slerp([0, 1], R.concatenate([quat_start, quat_goal]))
        quat_interp = slerp_obj([slerp_frac]).as_quat()[0]

        subgoal[self.irs_rrt_3d.quat_ind] = quat_interp[[3, 0, 1, 2]]

        return subgoal
    
    def calc_du_star_towards_q_lstsq(self, parent_node: Node, q: np.ndarray):
        # Compute least-squares solution.
        # NOTE(terry-suh): it is important to only do this on the submatrix
        # of B that has to do with u.

        idx_obj = [0, 1, 2, 3]

        du_star = np.linalg.lstsq(
            parent_node.Bhat[idx_obj, :],
            (q - parent_node.chat)[idx_obj],
            rcond=None,
        )[0]

        # Normalize least-squares solution.
        du_norm = np.linalg.norm(du_star)
        step_size = min(du_norm, self.rrt_params.stepsize)
        du_star = du_star / du_norm
        u_star = parent_node.ubar + step_size * du_star

        if self.rrt_params.enforce_robot_joint_limits:
            idx_robot = self.q_sim.get_q_a_indices_into_q()
            q_a_lb = self.q_lb[idx_robot]
            q_a_ub = self.q_ub[idx_robot]
            u_star = np.clip(u_star, q_a_lb, q_a_ub)

        return u_star - parent_node.ubar
    
    def extend_towards_q(self, parent_node: Node, q: np.array):
        """
        Extend towards a specified configuration q and return a new
        node,
        """
        regrasp = np.random.rand() < self.rrt_params.grasp_prob

        # breakpoint()
        parent_q_rpy = convert_state_quat_to_rpy(parent_node.q)
        if regrasp:
            x_next = self.contact_sampler.sample_contact(parent_q_rpy)
        else:
            du_star = self.calc_du_star_towards_q_lstsq(parent_node, q)
            u_star = parent_node.ubar + du_star
            x_next = self.q_sim.calc_dynamics(
                parent_q_rpy, u_star, self.sim_params
            )
        x_next = convert_state_rpy_to_quat(x_next)

        cost = 0.0

        child_node = IrsNode(x_next)
        child_node.subgoal = q

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.cost = cost

        if regrasp:
            edge.du = np.nan
            edge.u = np.nan
        else:
            edge.du = du_star
            edge.u = u_star

        return child_node, edge
    
    def calc_distance_batch_quat_diff(
        self, q_query: np.ndarray, n_nodes: int, is_q_u_only: bool
    ):
        # breakpoint()
        if is_q_u_only:
            q_query = q_query[self.q_u_indices_into_x]
        # B x n
        mu_batch = self.get_chat_matrix_up_to(n_nodes, is_q_u_only)
        metric_batch = np.abs(quat_angle_difference(
            q_query[self.irs_rrt_3d.quat_ind],
            mu_batch[:, self.irs_rrt_3d.quat_ind],
        ))

        return metric_batch
    
    def iterate(self):
        """
        Main method for iteration.
        """

        pbar = tqdm(total=self.max_size)

        while self.size < self.rrt_params.max_size:
            # 1. Sample a subgoal.
            rand_num = np.random.rand()
            # if self.cointoss_for_goal():
            if 0.0 <= rand_num < 0.3:
                subgoal = self.rrt_params.goal
            elif 0.3 <= rand_num < 0.95:
                subgoal = self.sample_subgoal_between_start_goal()
            else:
                subgoal = self.sample_random_subgoal()

            # 2. Sample closest node to subgoal
            parent_node = self.select_closest_node(
                subgoal, d_threshold=self.rrt_params.distance_threshold
            )
            if parent_node is None:
                continue
            # update progress only if a valid parent_node is chosen.

            # 3. Extend to subgoal.
            try:
                child_node, edge = self.extend(parent_node, subgoal)
            except RuntimeError:
                continue

            # 4. Attempt to rewire a candidate child node.
            if self.rrt_params.rewire:
                parent_node, child_node, edge = self.rewire(
                    parent_node, child_node
                )

            # 5. Register the new node to the graph.
            try:
                # Drawing every new node in meshcat seems to slow down
                #  tree building by quite a bit.
                self.add_node(child_node, draw_node=self.size % 3 == 0)
            except RuntimeError as e:
                print(e)
                continue
            pbar.update(1)

            child_node.value = parent_node.value + edge.cost
            self.add_edge(edge)

            # 6. Check for termination.
            if self.is_close_to_goal():
                self.goal_node_idx = child_node.id
                print("FOUND A PATH TO GOAL!!!!!")
                break

        pbar.close()
