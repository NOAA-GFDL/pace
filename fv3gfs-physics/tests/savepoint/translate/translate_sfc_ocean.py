from fv3gfs.physics.stencils.surface import sfc_ocean
from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py


class TranslateSfcOcean1(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "cm": {"serialname": "cd33", "sfc2d": True},
            "ch": {"serialname": "cdq33", "sfc2d": True},
            "chh": {"serialname": "chh33", "sfc2d": True},
            "cmm": {"serialname": "cmm33", "sfc2d": True},
            "ep": {"serialname": "ep1d33", "sfc2d": True},
            "evap": {"serialname": "evap33", "sfc2d": True},
            "flag_iter": {"serialname": "flag_iter", "sfc2d": True},
            "gflux": {"serialname": "gflx33", "sfc2d": True},
            "hflx": {"serialname": "hflx33", "sfc2d": True},
            "ps": {"serialname": "pgr", "sfc2d": True},
            "prsl1": {"serialname": "prsl_sfc", "sfc2d": True},
            "q1": {"serialname": "qgrs_sfc", "sfc2d": True},
            "qsurf": {"serialname": "qss33", "sfc2d": True},
            "t1": {"serialname": "tgrs_sfc", "sfc2d": True},
            "tskin": {"serialname": "tsfc33", "sfc2d": True},
            "wet": {"serialname": "wet", "sfc2d": True},
            "wind": {"serialname": "wind", "sfc2d": True},
            "prslki": {"serialname": "work3", "sfc2d": True},
        }
        self.out_vars = {
            "chh": self.in_vars["data_vars"]["chh"],
            "cmm": self.in_vars["data_vars"]["cmm"],
            "ep": self.in_vars["data_vars"]["ep"],
            "evap": self.in_vars["data_vars"]["evap"],
            "gflux": self.in_vars["data_vars"]["gflux"],
            "hflx": self.in_vars["data_vars"]["hflx"],
            "qsurf": self.in_vars["data_vars"]["qsurf"],
        }

    def compute(self, inputs):
        qsurf, cmm, chh, gflux, evap, hflx, ep = sfc_ocean(**inputs)
        output = {}
        output["qss33"] = qsurf
        output["cmm33"] = cmm
        output["chh33"] = chh
        output["gflx33"] = gflux
        output["evap33"] = evap
        output["hflx33"] = hflx
        output["ep1d33"] = ep
        ij = qsurf.shape[0] * qsurf.shape[1]
        for var in output.keys():
            output[var] = output[var].reshape(ij)
        return output


class TranslateSfcOcean2(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "cm": {"serialname": "cd33", "sfc2d": True},
            "ch": {"serialname": "cdq33", "sfc2d": True},
            "chh": {"serialname": "chh33", "sfc2d": True},
            "cmm": {"serialname": "cmm33", "sfc2d": True},
            "ep": {"serialname": "ep1d33", "sfc2d": True},
            "evap": {"serialname": "evap33", "sfc2d": True},
            "flag_iter": {"serialname": "flag_iter", "sfc2d": True},
            "gflux": {"serialname": "gflx33", "sfc2d": True},
            "hflx": {"serialname": "hflx33", "sfc2d": True},
            "ps": {"serialname": "pgr", "sfc2d": True},
            "prsl1": {"serialname": "prsl_sfc", "sfc2d": True},
            "q1": {"serialname": "qgrs_sfc", "sfc2d": True},
            "qsurf": {"serialname": "qss33", "sfc2d": True},
            "t1": {"serialname": "tgrs_sfc", "sfc2d": True},
            "tskin": {"serialname": "tsfc33", "sfc2d": True},
            "wet": {"serialname": "wet", "sfc2d": True},
            "wind": {"serialname": "wind", "sfc2d": True},
            "prslki": {"serialname": "work3", "sfc2d": True},
        }
        self.out_vars = {
            "chh": self.in_vars["data_vars"]["chh"],
            "cmm": self.in_vars["data_vars"]["cmm"],
            "ep": self.in_vars["data_vars"]["ep"],
            "evap": self.in_vars["data_vars"]["evap"],
            "gflux": self.in_vars["data_vars"]["gflux"],
            "hflx": self.in_vars["data_vars"]["hflx"],
            "qsurf": self.in_vars["data_vars"]["qsurf"],
        }

    def compute(self, inputs):
        qsurf, cmm, chh, gflux, evap, hflx, ep = sfc_ocean(**inputs)
        output = {}
        output["qss33"] = qsurf
        output["cmm33"] = cmm
        output["chh33"] = chh
        output["gflx33"] = gflux
        output["evap33"] = evap
        output["hflx33"] = hflx
        output["ep1d33"] = ep
        ij = qsurf.shape[0] * qsurf.shape[1]
        for var in output.keys():
            output[var] = output[var].reshape(ij)
        return output