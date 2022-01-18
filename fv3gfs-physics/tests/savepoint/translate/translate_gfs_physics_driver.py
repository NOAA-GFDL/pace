import copy

import pace.dsl.gt4py_utils as utils
import pace.util as util
from fv3gfs.physics.stencils.physics import Physics, PhysicsState
from pace.dsl.typing import Float
from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py
from pace.util.mpi import MPI


class TranslateGFSPhysicsDriver(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "qvapor": {"dycore": True},
            "qliquid": {"dycore": True},
            "qrain": {"dycore": True},
            "qsnow": {"dycore": True},
            "qice": {"dycore": True},
            "qgraupel": {"dycore": True},
            "qo3mr": {"dycore": True},
            "qsgs_tke": {"dycore": True},
            "qcld": {"dycore": True},
            "pt": {"dycore": True},
            "delp": {"dycore": True},
            "delz": {"dycore": True},
            "ua": {"dycore": True},
            "va": {"dycore": True},
            "w": {"dycore": True},
            "omga": {"dycore": True},
        }
        self.out_vars = {
            "gt0": {
                "serialname": "IPD_gt0",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "gu0": {
                "serialname": "IPD_gu0",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "gv0": {
                "serialname": "IPD_gv0",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qvapor": {
                "serialname": "IPD_qvapor",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qliquid": {
                "serialname": "IPD_qliquid",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qrain": {
                "serialname": "IPD_rain",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qice": {
                "serialname": "IPD_qice",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qsnow": {
                "serialname": "IPD_snow",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qgraupel": {
                "serialname": "IPD_qgraupel",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qcld": {
                "serialname": "IPD_qcld",
                "kend": namelist.npz - 1,
                "order": "F",
            },
        }
        self.namelist = namelist
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        storage = utils.make_storage_from_shape(
            self.grid_indexing.domain_full(add=(1, 1, 1)),
            origin=self.grid_indexing.origin_compute(),
            init=True,
            backend=self.stencil_factory.backend,
        )
        inputs["delprsi"] = copy.deepcopy(storage)
        inputs["phii"] = copy.deepcopy(storage)
        inputs["phil"] = copy.deepcopy(storage)
        inputs["dz"] = copy.deepcopy(storage)
        inputs["wmp"] = copy.deepcopy(storage)
        inputs["physics_updated_specific_humidity"] = copy.deepcopy(storage)
        inputs["physics_updated_qliquid"] = copy.deepcopy(storage)
        inputs["physics_updated_qrain"] = copy.deepcopy(storage)
        inputs["physics_updated_qsnow"] = copy.deepcopy(storage)
        inputs["physics_updated_qice"] = copy.deepcopy(storage)
        inputs["physics_updated_qgraupel"] = copy.deepcopy(storage)
        inputs["physics_updated_cloud_fraction"] = copy.deepcopy(storage)
        inputs["physics_updated_pt"] = copy.deepcopy(storage)
        inputs["physics_updated_ua"] = copy.deepcopy(storage)
        inputs["physics_updated_va"] = copy.deepcopy(storage)
        inputs["prsi"] = copy.deepcopy(storage)
        inputs["prsik"] = copy.deepcopy(storage)
        sizer = util.SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        quantity_factory = util.QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )
        active_packages = ["microphysics"]
        physics_state = PhysicsState(
            **inputs,
            quantity_factory=quantity_factory,
            active_packages=active_packages,
        )
        # make mock communicator, this is not used
        # but needs to be passed as type CubedSphereCommunicator
        comm = MPI.COMM_WORLD
        layout = [1, 1]
        partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
        communicator = util.CubedSphereCommunicator(comm, partitioner)
        # because it's not in the serialized data
        self.grid.grid_data.ptop = 300.0
        physics = Physics(
            self.stencil_factory,
            self.grid.grid_data,
            self.namelist,
            active_packages=active_packages,
        )
        physics._atmos_phys_driver_statein(
            physics_state.prsik,
            physics_state.phii,
            physics_state.prsi,
            physics_state.delz,
            physics_state.delp,
            physics_state.qvapor,
            physics_state.qliquid,
            physics_state.qrain,
            physics_state.qice,
            physics_state.qsnow,
            physics_state.qgraupel,
            physics_state.qo3mr,
            physics_state.qsgs_tke,
            physics_state.qcld,
            physics_state.pt,
            physics._dm3d,
        )
        physics._get_prs_fv3(
            physics_state.phii,
            physics_state.prsi,
            physics_state.pt,
            physics_state.qvapor,
            physics_state.delprsi,
            physics._del_gz,
        )
        # If PBL scheme is present, physics_state should be updated here
        physics._get_phi_fv3(
            physics_state.pt,
            physics_state.qvapor,
            physics._del_gz,
            physics_state.phii,
            physics_state.phil,
        )
        physics._prepare_microphysics(
            physics_state.dz,
            physics_state.phii,
            physics_state.wmp,
            physics_state.omga,
            physics_state.qvapor,
            physics_state.pt,
            physics_state.delp,
        )
        microph_state = physics_state.microphysics
        physics._microphysics(microph_state)
        # Fortran uses IPD interface, here we use physics_updated_<var>
        # to denote the updated field
        physics._update_physics_state_with_tendencies(
            physics_state.qvapor,
            physics_state.qliquid,
            physics_state.qrain,
            physics_state.qice,
            physics_state.qsnow,
            physics_state.qgraupel,
            physics_state.qcld,
            physics_state.pt,
            physics_state.ua,
            physics_state.va,
            microph_state.qv_dt,
            microph_state.ql_dt,
            microph_state.qr_dt,
            microph_state.qi_dt,
            microph_state.qs_dt,
            microph_state.qg_dt,
            microph_state.qa_dt,
            microph_state.pt_dt,
            microph_state.udt,
            microph_state.vdt,
            physics_state.physics_updated_specific_humidity,
            physics_state.physics_updated_qliquid,
            physics_state.physics_updated_qrain,
            physics_state.physics_updated_qice,
            physics_state.physics_updated_qsnow,
            physics_state.physics_updated_qgraupel,
            physics_state.physics_updated_cloud_fraction,
            physics_state.physics_updated_pt,
            physics_state.physics_updated_ua,
            physics_state.physics_updated_va,
            Float(physics._dt_atmos),
        )
        inputs["gt0"] = physics_state.physics_updated_pt
        inputs["gu0"] = physics_state.physics_updated_ua
        inputs["gv0"] = physics_state.physics_updated_va
        inputs["qvapor"] = physics_state.physics_updated_specific_humidity
        inputs["qliquid"] = physics_state.physics_updated_qliquid
        inputs["qrain"] = physics_state.physics_updated_qrain
        inputs["qice"] = physics_state.physics_updated_qice
        inputs["qsnow"] = physics_state.physics_updated_qsnow
        inputs["qgraupel"] = physics_state.physics_updated_qgraupel
        inputs["qcld"] = physics_state.physics_updated_cloud_fraction
        return self.slice_output(inputs)
