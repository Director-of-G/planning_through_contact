import numpy as np
from irs_rrt.irs_rrt import IrsRrt, IrsNode, IrsEdge
from irs_rrt.rrt_base import Node
from irs_rrt.rrt_params import IrsRrtParams
from pydrake.all import RollPitchYaw, Quaternion, RotationMatrix
from scipy.spatial.transform import Rotation as R
from irs_rrt.allegro_3d_rpy_helper import convert_state_quat_to_rpy, convert_state_rpy_to_quat

from qsim_cpp import QuasistaticSimulatorCpp
from qsim.simulator import QuasistaticSimulator


class IrsRrt3DRPY(IrsRrt):
    def __init__(
        self,
        rrt_params: IrsRrtParams,
        q_sim: QuasistaticSimulatorCpp,
        q_sim_py: QuasistaticSimulator,
    ):
        IrsRrt.__init__(
            self, rrt_params=rrt_params, q_sim=q_sim, q_sim_py=q_sim_py,
            use_rpy_reachable_set=True
        )

        self.qa_dim = self.dim_u
        self.quat_ind = [0, 1, 2, 3]
        self.q_u_indices_into_x = [0, 1, 2, 3]
        assert (self.rrt_params.global_metric[self.quat_ind] == 0).all()

    def sample_subgoal(self):
        # Sample translation
        rand_num = np.random.rand(16)
        subgoal = np.zeros(20)
        subgoal[-16:] = self.q_lb[-16:] + (self.q_ub[-16:] - self.q_lb[-16:]) * rand_num

        quat = R.random().as_quat()[[3, 0, 1, 2]]
        subgoal[self.quat_ind] = quat

        return subgoal

    def extend_towards_q(self, parent_node: Node, q: np.array):
        """
        Extend towards a specified configuration q and return a new
        node,
        """
        # Compute least-squares solution.
        du = np.linalg.lstsq(
            parent_node.Bhat, q - parent_node.chat, rcond=None
        )[0]

        # Normalize least-squares solution.
        du = du / np.linalg.norm(du)
        ustar = parent_node.ubar + self.rrt_params.stepsize * du

        # convert node (quat) to rpy
        parent_q = convert_state_quat_to_rpy(parent_node.q)

        xnext = self.q_sim.calc_dynamics(parent_q, ustar, self.sim_params)

        # convert fdyn result back to quat
        xnext = convert_state_rpy_to_quat(xnext)
        
        cost = self.reachable_set.calc_node_metric(
            parent_node.covinv, parent_node.mu, xnext
        )

        child_node = IrsNode(xnext)

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.du = self.rrt_params.stepsize * du
        edge.u = ustar
        edge.cost = cost

        return child_node, edge

    def compute_edge_cost(self, parent_q: Node, child_q: Node):
        error = parent_q - child_q
        cost = error @ self.metric_mat @ error

        parent_quat = R.from_quat(
            self.convert_quat_wxyz_to_xyzw(parent_q[self.quat_ind])
        )
        child_quat = R.from_quat(
            self.convert_quat_wxyz_to_xyzw(child_q[self.quat_ind])
        )
        quat_mul_diff = (child_quat * parent_quat.inv()).as_quat()
        cost += self.rrt_params.quat_metric * np.linalg.norm(quat_mul_diff[:-1])
        return cost

    def calc_distance_batch_global(
        self, q_query: np.ndarray, n_nodes: int, is_q_u_only: bool
    ):

        q_batch = self.get_q_matrix_up_to(n_nodes)

        if is_q_u_only:
            error_batch = (
                q_query[self.q_u_indices_into_x]
                - q_batch[:, self.q_u_indices_into_x]
            )
            metric_mat = np.diag(
                self.rrt_params.global_metric[self.q_u_indices_into_x]
            )
        else:
            error_batch = q_query - q_batch
            metric_mat = np.diag(self.rrt_params.global_metric)

        intsum = np.einsum("Bi,ij->Bj", error_batch, metric_mat)
        metric_batch = np.einsum("Bi,Bi->B", intsum, error_batch)

        # scipy accepts (x, y, z, w)
        q_query_quat = R.from_quat(
            self.convert_quat_wxyz_to_xyzw(q_query[self.quat_ind])
        )
        quat_batch = R.from_quat(
            self.convert_quat_wxyz_to_xyzw(
                q_batch[:, self.quat_ind], batch_mode=True
            )
        )
        quat_mul_diff = (quat_batch * q_query_quat.inv()).as_quat()
        metric_batch += self.rrt_params.quat_metric * np.linalg.norm(
            quat_mul_diff[:, :-1], axis=1
        )

        return metric_batch

    def convert_quat_wxyz_to_xyzw(self, q, batch_mode=False):
        if batch_mode:
            return q[:, [1, 2, 3, 0]]
        else:
            return q[[1, 2, 3, 0]]

    def convert_quat_xyzw_to_wxyz(self, q, batch_mode=False):
        if batch_mode:
            return q[:, [3, 0, 1, 2]]
        else:
            return q[[3, 0, 1, 2]]
