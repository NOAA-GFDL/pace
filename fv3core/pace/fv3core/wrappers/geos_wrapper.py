import enum
import logging
import os
from datetime import timedelta
from typing import Dict, List, Tuple

import f90nml
import numpy as np
from gt4py.cartesian.config import build_settings as gt_build_settings
from mpi4py import MPI

import pace.util
from pace import fv3core
from pace.driver.performance.collector import PerformanceCollector
from pace.dsl.dace import orchestrate
from pace.dsl.dace.build import set_distributed_caches
from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from pace.dsl.gt4py_utils import is_gpu_backend
from pace.dsl.typing import floating_point_precision
from pace.util._optional_imports import cupy as cp
from pace.util.logging import pace_log


class StencilBackendCompilerOverride:
    """Override the Pace global stencil JIT to allow for 9-rank build
    on any setup.

    This is a workaround that requires to know _exactly_ when build is happening.
    Using this as a context manager, we leverage the DaCe build system to override
    the name and build the 9 codepaths required- while every other rank wait.

    This should be removed when we refactor the GT JIT to distribute building
    much more efficiently
    """

    def __init__(self, comm: MPI.Intracomm, config: DaceConfig):
        self.comm = comm
        self.config = config

        # Orchestration or mono-node is not concerned
        self.no_op = self.config.is_dace_orchestrated() or self.comm.Get_size() == 1

        # We abuse the DaCe build system
        if not self.no_op:
            config._orchestrate = DaCeOrchestration.Build
            set_distributed_caches(config)
            config._orchestrate = DaCeOrchestration.Python

        # We remove warnings from the stencils compiling when in critical and/or
        # error
        if pace_log.level > logging.WARNING:
            gt_build_settings["extra_compile_args"]["cxx"].append("-w")
            gt_build_settings["extra_compile_args"]["cuda"].append("-w")

    def __enter__(self):
        if self.no_op:
            return
        if self.config.do_compile:
            pace_log.info(f"Stencil backend compiles on {self.comm.Get_rank()}")
        else:
            pace_log.info(f"Stencil backend waits on {self.comm.Get_rank()}")
            self.comm.Barrier()

    def __exit__(self, type, value, traceback):
        if self.no_op:
            return
        if not self.config.do_compile:
            pace_log.info(f"Stencil backend read cache on {self.comm.Get_rank()}")
        else:
            pace_log.info(f"Stencil backend compiled on {self.comm.Get_rank()}")
            self.comm.Barrier()


@enum.unique
class MemorySpace(enum.Enum):
    HOST = 0
    DEVICE = 1


class GeosDycoreWrapper:
    """
    Provides an interface for the Geos model to access the Pace dycore.
    Takes numpy arrays as inputs, returns a dictionary of numpy arrays as outputs
    """

    def __init__(
        self,
        namelist: f90nml.Namelist,
        bdt: int,
        comm: pace.util.Comm,
        backend: str,
        fortran_mem_space: MemorySpace = MemorySpace.HOST,
    ):
        # Look for an override to run on a single node
        gtfv3_single_rank_override = int(os.getenv("GTFV3_SINGLE_RANK_OVERRIDE", -1))
        if gtfv3_single_rank_override >= 0:
            comm = pace.util.NullComm(gtfv3_single_rank_override, 6, 42)

        # Make a custom performance collector for the GEOS wrapper
        self.perf_collector = PerformanceCollector("GEOS wrapper", comm)

        self.backend = backend
        self.namelist = namelist
        self.dycore_config = fv3core.DynamicalCoreConfig.from_f90nml(self.namelist)
        self.dycore_config.dt_atmos = bdt
        assert self.dycore_config.dt_atmos != 0

        self.layout = self.dycore_config.layout
        partitioner = pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(self.layout)
        )
        self.communicator = pace.util.CubedSphereCommunicator(
            comm,
            partitioner,
            timer=self.perf_collector.timestep_timer,
        )

        sizer = pace.util.SubtileGridSizer.from_namelist(
            self.namelist, partitioner.tile, self.communicator.tile.rank
        )
        quantity_factory = pace.util.QuantityFactory.from_backend(
            sizer=sizer, backend=backend
        )

        # set up the metric terms and grid data
        metric_terms = pace.util.grid.MetricTerms(
            quantity_factory=quantity_factory, communicator=self.communicator
        )
        grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)

        stencil_config = pace.dsl.stencil.StencilConfig(
            compilation_config=pace.dsl.stencil.CompilationConfig(
                backend=backend, rebuild=False, validate_args=False
            ),
        )

        # Build a DaCeConfig for orchestration.
        # This and all orchestration code are transparent when outside
        # configuration deactivate orchestration
        stencil_config.dace_config = DaceConfig(
            communicator=self.communicator,
            backend=stencil_config.backend,
            tile_nx=self.dycore_config.npx,
            tile_nz=self.dycore_config.npz,
        )
        self._is_orchestrated = stencil_config.dace_config.is_dace_orchestrated()

        # Orchestrate all code called from this function
        orchestrate(
            obj=self,
            config=stencil_config.dace_config,
            method_to_orchestrate="_critical_path",
        )

        self._grid_indexing = pace.dsl.stencil.GridIndexing.from_sizer_and_communicator(
            sizer=sizer, comm=self.communicator
        )
        stencil_factory = pace.dsl.StencilFactory(
            config=stencil_config, grid_indexing=self._grid_indexing
        )

        self.dycore_state = fv3core.DycoreState.init_zeros(
            quantity_factory=quantity_factory
        )
        self.dycore_state.bdt = self.dycore_config.dt_atmos

        damping_coefficients = pace.util.grid.DampingCoefficients.new_from_metric_terms(
            metric_terms
        )

        with StencilBackendCompilerOverride(MPI.COMM_WORLD, stencil_config.dace_config):
            self.dynamical_core = fv3core.DynamicalCore(
                comm=self.communicator,
                grid_data=grid_data,
                stencil_factory=stencil_factory,
                quantity_factory=quantity_factory,
                damping_coefficients=damping_coefficients,
                config=self.dycore_config,
                timestep=timedelta(seconds=self.dycore_state.bdt),
                phis=self.dycore_state.phis,
                state=self.dycore_state,
            )

        self._fortran_mem_space = fortran_mem_space
        self._pace_mem_space = (
            MemorySpace.DEVICE if is_gpu_backend(backend) else MemorySpace.HOST
        )

        self.output_dict: Dict[str, np.ndarray] = {}
        self._allocate_output_dir()

        # Feedback information
        device_ordinal_info = (
            f"  Device PCI bus id: {cp.cuda.Device(0).pci_bus_id}\n"
            if is_gpu_backend(backend)
            else "N/A"
        )
        MPS_pipe_directory = os.getenv("CUDA_MPS_PIPE_DIRECTORY", None)
        MPS_is_on = (
            MPS_pipe_directory is not None
            and is_gpu_backend(backend)
            and os.path.exists(f"{MPS_pipe_directory}/log")
        )
        pace_log.info(
            "Pace GEOS wrapper initialized: \n"
            f"             dt : {self.dycore_state.bdt}\n"
            f"         bridge : {self._fortran_mem_space} > {self._pace_mem_space}\n"
            f"        backend : {backend}\n"
            f"          float : {floating_point_precision()}bit"
            f"  orchestration : {self._is_orchestrated}\n"
            f"          sizer : {sizer.nx}x{sizer.ny}x{sizer.nz}"
            f"(halo: {sizer.n_halo})\n"
            f"     Device ord : {device_ordinal_info}\n"
            f"     Nvidia MPS : {MPS_is_on}"
        )

    def _critical_path(self):
        """Top-level orchestration function"""
        with self.perf_collector.timestep_timer.clock("step_dynamics"):
            self.dynamical_core.step_dynamics(
                state=self.dycore_state,
                timer=self.perf_collector.timestep_timer,
            )

    def __call__(
        self,
        timings: Dict[str, List[float]],
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        delz: np.ndarray,
        pt: np.ndarray,
        delp: np.ndarray,
        q: np.ndarray,
        ps: np.ndarray,
        pe: np.ndarray,
        pk: np.ndarray,
        peln: np.ndarray,
        pkz: np.ndarray,
        phis: np.ndarray,
        q_con: np.ndarray,
        omga: np.ndarray,
        ua: np.ndarray,
        va: np.ndarray,
        uc: np.ndarray,
        vc: np.ndarray,
        mfxd: np.ndarray,
        mfyd: np.ndarray,
        cxd: np.ndarray,
        cyd: np.ndarray,
        diss_estd: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
        with self.perf_collector.timestep_timer.clock("numpy-to-dycore"):
            self.dycore_state = self._put_fortran_data_in_dycore(
                u,
                v,
                w,
                delz,
                pt,
                delp,
                q,
                ps,
                pe,
                pk,
                peln,
                pkz,
                phis,
                q_con,
                omga,
                ua,
                va,
                uc,
                vc,
                mfxd,
                mfyd,
                cxd,
                cyd,
                diss_estd,
            )

        # Enter orchestrated code - if applicable
        self._critical_path()

        with self.perf_collector.timestep_timer.clock("dycore-to-numpy"):
            self.output_dict = self._prep_outputs_for_geos()

        # Collect performance of the timestep and write a json file for rank 0
        self.perf_collector.collect_performance()
        for k, v in self.perf_collector.times_per_step[0].items():
            if k not in timings.keys():
                timings[k] = [v]
            else:
                timings[k].append(v)
        self.perf_collector.clear()

        return self.output_dict, timings

    def _put_fortran_data_in_dycore(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        delz: np.ndarray,
        pt: np.ndarray,
        delp: np.ndarray,
        q: np.ndarray,
        ps: np.ndarray,
        pe: np.ndarray,
        pk: np.ndarray,
        peln: np.ndarray,
        pkz: np.ndarray,
        phis: np.ndarray,
        q_con: np.ndarray,
        omga: np.ndarray,
        ua: np.ndarray,
        va: np.ndarray,
        uc: np.ndarray,
        vc: np.ndarray,
        mfxd: np.ndarray,
        mfyd: np.ndarray,
        cxd: np.ndarray,
        cyd: np.ndarray,
        diss_estd: np.ndarray,
    ) -> fv3core.DycoreState:
        isc = self._grid_indexing.isc
        jsc = self._grid_indexing.jsc
        iec = self._grid_indexing.iec + 1
        jec = self._grid_indexing.jec + 1

        state = self.dycore_state

        # Assign compute domain:
        pace.util.utils.safe_assign_array(state.u.view[:], u[isc:iec, jsc : jec + 1, :])
        pace.util.utils.safe_assign_array(state.v.view[:], v[isc : iec + 1, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.w.view[:], w[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.ua.view[:], ua[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.va.view[:], va[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(
            state.uc.view[:], uc[isc : iec + 1, jsc:jec, :]
        )
        pace.util.utils.safe_assign_array(
            state.vc.view[:], vc[isc:iec, jsc : jec + 1, :]
        )

        pace.util.utils.safe_assign_array(state.delz.view[:], delz[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.pt.view[:], pt[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.delp.view[:], delp[isc:iec, jsc:jec, :])

        pace.util.utils.safe_assign_array(state.mfxd.view[:], mfxd)
        pace.util.utils.safe_assign_array(state.mfyd.view[:], mfyd)
        pace.util.utils.safe_assign_array(state.cxd.view[:], cxd[:, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.cyd.view[:], cyd[isc:iec, :, :])

        pace.util.utils.safe_assign_array(state.ps.view[:], ps[isc:iec, jsc:jec])
        pace.util.utils.safe_assign_array(
            state.pe.data[isc - 1 : iec + 1, jsc - 1 : jec + 1, :], pe
        )
        pace.util.utils.safe_assign_array(state.pk.view[:], pk)
        pace.util.utils.safe_assign_array(state.peln.view[:], peln)
        pace.util.utils.safe_assign_array(state.pkz.view[:], pkz)
        pace.util.utils.safe_assign_array(state.phis.view[:], phis[isc:iec, jsc:jec])
        pace.util.utils.safe_assign_array(
            state.q_con.view[:], q_con[isc:iec, jsc:jec, :]
        )
        pace.util.utils.safe_assign_array(state.omga.view[:], omga[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(
            state.diss_estd.view[:], diss_estd[isc:iec, jsc:jec, :]
        )

        # tracer quantities should be a 4d array in order:
        # vapor, liquid, ice, rain, snow, graupel, cloud
        pace.util.utils.safe_assign_array(
            state.qvapor.view[:], q[isc:iec, jsc:jec, :, 0]
        )
        pace.util.utils.safe_assign_array(
            state.qliquid.view[:], q[isc:iec, jsc:jec, :, 1]
        )
        pace.util.utils.safe_assign_array(state.qice.view[:], q[isc:iec, jsc:jec, :, 2])
        pace.util.utils.safe_assign_array(
            state.qrain.view[:], q[isc:iec, jsc:jec, :, 3]
        )
        pace.util.utils.safe_assign_array(
            state.qsnow.view[:], q[isc:iec, jsc:jec, :, 4]
        )
        pace.util.utils.safe_assign_array(
            state.qgraupel.view[:], q[isc:iec, jsc:jec, :, 5]
        )
        pace.util.utils.safe_assign_array(state.qcld.view[:], q[isc:iec, jsc:jec, :, 6])

        return state

    def _prep_outputs_for_geos(self) -> Dict[str, np.ndarray]:
        output_dict = self.output_dict
        isc = self._grid_indexing.isc
        jsc = self._grid_indexing.jsc
        iec = self._grid_indexing.iec + 1
        jec = self._grid_indexing.jec + 1

        if self._fortran_mem_space != self._pace_mem_space:
            pace.util.utils.safe_assign_array(
                output_dict["u"], self.dycore_state.u.data[:-1, :, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["v"], self.dycore_state.v.data[:, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["w"], self.dycore_state.w.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["ua"], self.dycore_state.ua.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["va"], self.dycore_state.va.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["uc"], self.dycore_state.uc.data[:, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["vc"], self.dycore_state.vc.data[:-1, :, :-1]
            )

            pace.util.utils.safe_assign_array(
                output_dict["delz"], self.dycore_state.delz.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["pt"], self.dycore_state.pt.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["delp"], self.dycore_state.delp.data[:-1, :-1, :-1]
            )

            pace.util.utils.safe_assign_array(
                output_dict["mfxd"],
                self.dycore_state.mfxd.data[isc : iec + 1, jsc:jec, :-1],
            )
            pace.util.utils.safe_assign_array(
                output_dict["mfyd"],
                self.dycore_state.mfyd.data[isc:iec, jsc : jec + 1, :-1],
            )
            pace.util.utils.safe_assign_array(
                output_dict["cxd"], self.dycore_state.cxd.data[isc : iec + 1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["cyd"], self.dycore_state.cyd.data[:-1, jsc : jec + 1, :-1]
            )

            pace.util.utils.safe_assign_array(
                output_dict["ps"], self.dycore_state.ps.data[:-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["pe"],
                self.dycore_state.pe.data[isc - 1 : iec + 1, jsc - 1 : jec + 1, :],
            )
            pace.util.utils.safe_assign_array(
                output_dict["pk"], self.dycore_state.pk.data[isc:iec, jsc:jec, :]
            )
            pace.util.utils.safe_assign_array(
                output_dict["peln"], self.dycore_state.peln.data[isc:iec, jsc:jec, :]
            )
            pace.util.utils.safe_assign_array(
                output_dict["pkz"], self.dycore_state.pkz.data[isc:iec, jsc:jec, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["phis"], self.dycore_state.phis.data[:-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["q_con"], self.dycore_state.q_con.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["omga"], self.dycore_state.omga.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["diss_estd"],
                self.dycore_state.diss_estd.data[:-1, :-1, :-1],
            )

            pace.util.utils.safe_assign_array(
                output_dict["qvapor"], self.dycore_state.qvapor.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["qliquid"], self.dycore_state.qliquid.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["qice"], self.dycore_state.qice.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["qrain"], self.dycore_state.qrain.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["qsnow"], self.dycore_state.qsnow.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["qgraupel"], self.dycore_state.qgraupel.data[:-1, :-1, :-1]
            )
            pace.util.utils.safe_assign_array(
                output_dict["qcld"], self.dycore_state.qcld.data[:-1, :-1, :-1]
            )
        else:
            output_dict["u"] = self.dycore_state.u.data[:-1, :, :-1]
            output_dict["v"] = self.dycore_state.v.data[:, :-1, :-1]
            output_dict["w"] = self.dycore_state.w.data[:-1, :-1, :-1]
            output_dict["ua"] = self.dycore_state.ua.data[:-1, :-1, :-1]
            output_dict["va"] = self.dycore_state.va.data[:-1, :-1, :-1]
            output_dict["uc"] = self.dycore_state.uc.data[:, :-1, :-1]
            output_dict["vc"] = self.dycore_state.vc.data[:-1, :, :-1]
            output_dict["delz"] = self.dycore_state.delz.data[:-1, :-1, :-1]
            output_dict["pt"] = self.dycore_state.pt.data[:-1, :-1, :-1]
            output_dict["delp"] = self.dycore_state.delp.data[:-1, :-1, :-1]
            output_dict["mfxd"] = self.dycore_state.mfxd.data[
                isc : iec + 1, jsc:jec, :-1
            ]
            output_dict["mfyd"] = self.dycore_state.mfyd.data[
                isc:iec, jsc : jec + 1, :-1
            ]
            output_dict["cxd"] = self.dycore_state.cxd.data[isc : iec + 1, :-1, :-1]
            output_dict["cyd"] = self.dycore_state.cyd.data[:-1, jsc : jec + 1, :-1]
            output_dict["ps"] = self.dycore_state.ps.data[:-1, :-1]
            output_dict["pe"] = self.dycore_state.pe.data[
                isc - 1 : iec + 1, jsc - 1 : jec + 1, :
            ]
            output_dict["pk"] = self.dycore_state.pk.data[isc:iec, jsc:jec, :]
            output_dict["peln"] = self.dycore_state.peln.data[isc:iec, jsc:jec, :]
            output_dict["pkz"] = self.dycore_state.pkz.data[isc:iec, jsc:jec, :-1]
            output_dict["phis"] = self.dycore_state.phis.data[:-1, :-1]
            output_dict["q_con"] = self.dycore_state.q_con.data[:-1, :-1, :-1]
            output_dict["omga"] = self.dycore_state.omga.data[:-1, :-1, :-1]
            output_dict["diss_estd"] = self.dycore_state.diss_estd.data[:-1, :-1, :-1]
            output_dict["qvapor"] = self.dycore_state.qvapor.data[:-1, :-1, :-1]
            output_dict["qliquid"] = self.dycore_state.qliquid.data[:-1, :-1, :-1]
            output_dict["qice"] = self.dycore_state.qice.data[:-1, :-1, :-1]
            output_dict["qrain"] = self.dycore_state.qrain.data[:-1, :-1, :-1]
            output_dict["qsnow"] = self.dycore_state.qsnow.data[:-1, :-1, :-1]
            output_dict["qgraupel"] = self.dycore_state.qgraupel.data[:-1, :-1, :-1]
            output_dict["qcld"] = self.dycore_state.qcld.data[:-1, :-1, :-1]

        return output_dict

    def _allocate_output_dir(self):
        if self._fortran_mem_space != self._pace_mem_space:
            nhalo = self._grid_indexing.n_halo
            shape_centered = self._grid_indexing.domain_full(add=(0, 0, 0))
            shape_x_interface = self._grid_indexing.domain_full(add=(1, 0, 0))
            shape_y_interface = self._grid_indexing.domain_full(add=(0, 1, 0))
            shape_z_interface = self._grid_indexing.domain_full(add=(0, 0, 1))
            shape_2d = shape_centered[:-1]

            self.output_dict["u"] = np.empty((shape_y_interface))
            self.output_dict["v"] = np.empty((shape_x_interface))
            self.output_dict["w"] = np.empty((shape_centered))
            self.output_dict["ua"] = np.empty((shape_centered))
            self.output_dict["va"] = np.empty((shape_centered))
            self.output_dict["uc"] = np.empty((shape_x_interface))
            self.output_dict["vc"] = np.empty((shape_y_interface))

            self.output_dict["delz"] = np.empty((shape_centered))
            self.output_dict["pt"] = np.empty((shape_centered))
            self.output_dict["delp"] = np.empty((shape_centered))

            self.output_dict["mfxd"] = np.empty(
                (self._grid_indexing.domain_full(add=(1 - 2 * nhalo, -2 * nhalo, 0)))
            )
            self.output_dict["mfyd"] = np.empty(
                (self._grid_indexing.domain_full(add=(-2 * nhalo, 1 - 2 * nhalo, 0)))
            )
            self.output_dict["cxd"] = np.empty(
                (self._grid_indexing.domain_full(add=(1 - 2 * nhalo, 0, 0)))
            )
            self.output_dict["cyd"] = np.empty(
                (self._grid_indexing.domain_full(add=(0, 1 - 2 * nhalo, 0)))
            )

            self.output_dict["ps"] = np.empty((shape_2d))
            self.output_dict["pe"] = np.empty(
                (self._grid_indexing.domain_full(add=(2 - 2 * nhalo, 2 - 2 * nhalo, 1)))
            )
            self.output_dict["pk"] = np.empty(
                (self._grid_indexing.domain_full(add=(-2 * nhalo, -2 * nhalo, 1)))
            )
            self.output_dict["peln"] = np.empty(
                (self._grid_indexing.domain_full(add=(-2 * nhalo, -2 * nhalo, 1)))
            )
            self.output_dict["pkz"] = np.empty(
                (self._grid_indexing.domain_full(add=(-2 * nhalo, -2 * nhalo, 0)))
            )
            self.output_dict["phis"] = np.empty((shape_2d))
            self.output_dict["q_con"] = np.empty((shape_centered))
            self.output_dict["omga"] = np.empty((shape_centered))
            self.output_dict["diss_estd"] = np.empty((shape_centered))

            self.output_dict["qvapor"] = np.empty((shape_centered))
            self.output_dict["qliquid"] = np.empty((shape_centered))
            self.output_dict["qice"] = np.empty((shape_centered))
            self.output_dict["qrain"] = np.empty((shape_centered))
            self.output_dict["qsnow"] = np.empty((shape_centered))
            self.output_dict["qgraupel"] = np.empty((shape_centered))
            self.output_dict["qcld"] = np.empty((shape_centered))
        else:
            self.output_dict["u"] = None
            self.output_dict["v"] = None
            self.output_dict["w"] = None
            self.output_dict["ua"] = None
            self.output_dict["va"] = None
            self.output_dict["uc"] = None
            self.output_dict["vc"] = None
            self.output_dict["delz"] = None
            self.output_dict["pt"] = None
            self.output_dict["delp"] = None
            self.output_dict["mfxd"] = None
            self.output_dict["mfyd"] = None
            self.output_dict["cxd"] = None
            self.output_dict["cyd"] = None
            self.output_dict["ps"] = None
            self.output_dict["pe"] = None
            self.output_dict["pk"] = None
            self.output_dict["peln"] = None
            self.output_dict["pkz"] = None
            self.output_dict["phis"] = None
            self.output_dict["q_con"] = None
            self.output_dict["omga"] = None
            self.output_dict["diss_estd"] = None
            self.output_dict["qvapor"] = None
            self.output_dict["qliquid"] = None
            self.output_dict["qice"] = None
            self.output_dict["qrain"] = None
            self.output_dict["qsnow"] = None
            self.output_dict["qgraupel"] = None
            self.output_dict["qcld"] = None
