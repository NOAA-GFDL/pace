import unittest.mock

import pytest

from ndsl import QuantityFactory, SubtileGridSizer
from pace.driver import DiagnosticsConfig, MonitorDiagnostics, NullDiagnostics, ZSelect
from pyFV3 import DycoreState


def test_returns_null_diagnostics_if_no_path_given():
    config = DiagnosticsConfig(path=None, names=[])
    assert isinstance(
        config.diagnostics_factory(unittest.mock.MagicMock()),
        NullDiagnostics,
    )


def test_returns_monitor_diagnostics_if_path_given(tmpdir):
    config = DiagnosticsConfig(
        path=tmpdir,
        names=["foo"],
        derived_names=["bar"],
        z_select=[ZSelect(level=0, names=["foo"])],
    )
    result = config.diagnostics_factory(unittest.mock.MagicMock())
    assert isinstance(result, MonitorDiagnostics)


def test_raises_if_names_given_but_no_path():
    with pytest.raises(ValueError):
        DiagnosticsConfig(path=None, names=["foo"])

    with pytest.raises(ValueError):
        DiagnosticsConfig(path=None, derived_names=["foo"])


def test_zselect_raises_error_if_not_3d(tmpdir):
    with pytest.raises(AssertionError):
        config = DiagnosticsConfig(
            path=tmpdir,
            z_select=[ZSelect(level=0, names=["phis"])],
        )
        result = config.diagnostics_factory(unittest.mock.MagicMock())
        quantity_factory = QuantityFactory.from_backend(
            sizer=SubtileGridSizer(nx=12, ny=12, nz=79, n_halo=3, extra_dim_lengths={}),
            backend="numpy",
        )
        state = DycoreState.init_zeros(quantity_factory)
        result.z_select[0].select_data(state)


def test_zselect_raises_error_if_3rd_dim_not_z(tmpdir):
    with pytest.raises(ValueError):
        config = DiagnosticsConfig(
            path=tmpdir,
            z_select=[ZSelect(level=0, names=["foo"])],
        )
        result = config.diagnostics_factory(unittest.mock.MagicMock())
        quantity_factory = QuantityFactory.from_backend(
            sizer=SubtileGridSizer(nx=12, ny=12, nz=79, n_halo=3, extra_dim_lengths={}),
            backend="numpy",
        )
        state = DycoreState.init_zeros(quantity_factory)
        foo = quantity_factory.zeros(dims=["z", "x", "y"], units="-")
        state.foo = foo
        result.z_select[0].select_data(state)
