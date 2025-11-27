from typing import Dict
import copy

import numpy as np
import networkx as nx
from irs_rrt.rrt_params import IrsRrtParams
from pydrake.all import AngleAxis, Quaternion, RotationMatrix
from qsim.simulator import (
    QuasistaticSimulator,
    QuasistaticSimParameters,
    GradientMode,
    ForwardDynamicsMode,
)
from irs_rrt.allegro_3d_rpy_helper import (
    convert_state_rpy_to_quat,
    convert_state_quat_to_rpy,
    convert_Bhat_rpy_to_quat
)
from qsim.parser import QuasistaticParser
from qsim_cpp import QuasistaticSimulatorCpp

from irs_rrt.reachable_set import ReachableSet
from irs_mpc2.irs_mpc_params import (
    kSmoothingMode2ForwardDynamicsModeMap,
    kNoSmoothingModes,
    k0RandomizedSmoothingModes,
    k1RandomizedSmoothingModes,
    kAnalyticSmoothingModes,
)


class ReachableSet3DRPY(ReachableSet):
    """
    Computation class that computes parameters and metrics of reachable sets.
    """

    def __init__(
        self,
        q_sim: QuasistaticSimulatorCpp,
        rrt_params: IrsRrtParams,
        sim_params: QuasistaticSimParameters,
    ):
        super().__init__(q_sim, rrt_params, sim_params, use_rpy_reachable_set=True)

    def calc_bundled_Bc_analytic(self, q, ubar):
        assert self.rrt_params.smoothing_mode in kAnalyticSmoothingModes
        self.sim_params.gradient_mode = GradientMode.kBOnly
        self.sim_params.forward_mode = kSmoothingMode2ForwardDynamicsModeMap[
            self.rrt_params.smoothing_mode
        ]
        
        q_rpy = convert_state_quat_to_rpy(q)

        q_next = self.q_sim.calc_dynamics(
            q=q_rpy, u=ubar, sim_params=self.sim_params
        )

        q_next = convert_state_rpy_to_quat(q_next)

        Bhat = self.q_sim.get_Dq_nextDqa_cmd()
        Bhat = convert_Bhat_rpy_to_quat(Bhat, q)

        return Bhat, q_next
