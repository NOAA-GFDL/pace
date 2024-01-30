History
=======

latest
------

- Added `fill_for_translate_test` to MetricTerms to fill fields with NaNs only when required for testing
- Added `init_cartesian` method to MetricTerms to handle grid generation for orthogonal grids
- Added `from_layout` and `size` methods to TileCommunicator and Communicator
- Added `__init__` and `total_ranks` abstract methods to Partitioner
- Added `grid_type` to MetricTerms and DriverGridData
- Added `dx_const`, `dy_const`, `deglat`, and `u_max` namelist settings for doubly-periodic grids
- Added `dx_const`, `dy_const`, and `deglat` to grid generation code for doubly-periodic grids
- Added f32 support to halo exchange data transformation
- Use one single logger, from logging.py
- Removed hard-coded values of `ak` and `bk` arrays and added in the feature to read in `ak` and `bk` values
  from a NetCDF file to compute the `eta` and `eta_v` values.

v0.10.0
-------

Major changes:

- Added the following attributes/methods to Communicator: `tile`, `halo_update`, `boundaries`, `start_halo_update`, `vector_halo_update`, `start_vector_halo_update`, `synchronize_vector_interfaces`, `start_synchronize_vector_interfaces`, `get_scalar_halo_updater`, and `get_vector_halo_updater`
- Added Checkpointer and NullCheckpointer classes
- Added SnapshotCheckpointer
- Comm and Request abstract base classes are added to the top level
- Added the following attributes/methods to the Comm abstract base classes: `allreduce`, `allgather`
- Added classes `Threshold`, `ThresholdCalibrationCheckpointer`, `ValidationCheckpointer`, and `SavepointThresholds`
- Added `get_fs` as publicly-available function
- Legacy restart routines can now load restart data from any fsspec-supported filesystem
- Legacy restart routines will raise an exception if no restart files are present instead of loading an empty state
- Added NetCDFMonitor for saving the global state in time-chunked NetCDF files

Minor changes:

- Deleted deprecated `finish_halo_update` method from CubedSphereCommunicator
- fixed a bug in `pace.util.grid` where `_reduce_global_area_minmaxes` would use local values instead of the gathered ones
- Added .cleanup() method to ZarrMonitor, used only for API compatibility with NetCDFMonitor and does nothing
- ZarrMonitor.partitioner may now be any Partitioner and not just a CubedSpherePartitioner
- Quantity no longer has a `storage` attribute - the ndarray is directly accessible through the `data` attribute.

Minor changes:

- Fixed a bug in normalize_vector(xyz) in `pace.util.grid.gnomonic` where it would divide the input by cells-per-tile, where it should not.
- Refactored `pace.util.grid.helper` so that `HorizontalGridData`, `VerticalGridData`, `ContravariantGridData` and `AngleGridData` have their own `new_from_metric_terms` class methods, and `GridData` calls those in its own method definition.
- Added `stretch_transformation` to `pace.util.grid` - stretches the grid as needed for refinement, tropical test case.

v0.9.0
------

Major changes:

- Modified `pace.util.Quantity.transpose` to retain attributes, and loosened `pace.util.ZarrMonitor.store` requirements on attribute consistency, both to ease fv3net integration issues not addressed in v0.8.0

v0.8.0
------

Major changes:

- Changed `ZarrMonitor.store` behavior to allow passing quantities with different dimension orders
- Added `CachingCommWriter` which wraps a `Comm` object and can be serialized to a file-like object with a `.dump` method
- Added `CachingCommReader` which can be loaded from the dump output of `CachingCommWriter` and replays its communication in the order it occurred.
- `NullComm` is now public api in `pace-util`
- Deleted deprecated `finish_vector_halo_update` method from `CubedSphereCommunicator`
- Renamed DummyComm to LocalComm, and added support for message tags. The DummyComm symbol is still in place for backwards compatibility, but points to LocalComm
- added error in CubedSphereCommunicator init if given a communicator with a size not equal to the total ranks of the given partitioner
- `subtile_extent` method of Partitioner classes now takes in a required `rank` argument
- TilePartitioner has a new `edge_interior_ratio` argument which defaults to 1.0, and lets the user specify the relative 1-dimensional extent of the compute domains of ranks on tile edges and corners relative to ranks on the tile interior. In all cases, the closest valid value will be used, which enables some previously invalid configurations (e.g. C128 on a 3 by 3 layout will use the closest valid edge_interior_ratio to 1.0)

Minor changes:

- The `split_cartesian_into_storages` method is moved out of pace-util, as it is more generally used, and now lives in pace.dsl.gt4py_utils
- created `DriverGridData.new_from_grid_variables` class method to initialize from grid variable data
- updated QuantityFactory to accept the more generic GridSizer class on initialization
- added `sizer` as public attribute on QuantityFactory
- added `Namelist` class to initialize namelist files used in fv3gfs-fortran
- added `CubedSphereCommunicator.from_layout` constructor method
- added support for built-in `datetime` in ZarrMonitor
- `edge_interior_ratio` is now an optional argument of `tile_extent_from_rank_metadata`
- added support for writing constant data (written once, does not change with time) in ZarrMonitor

v0.7.0
------

Major changes:

- Renamed package from fv3gfs-util to pace-util
- Added NullTimer to use for default Timer value, it is a disabled timer which cannot be enabled (raises NotImplementedError)
- Added pace.util.grid, keeping symbols out of top level as they are still unstable
- Added HaloUpdater and associated code, which compiles halo packing for more efficient halo updates
- Added physical constants to pace.util.constants

Minor changes:

- Added method set_extra_dim_lengths to QuantityFactory

Fixes:

- Fixed bug where ZarrMonitor depended on dict `.items()` always returning items in the same order

Other changes may exist in this version, as we temporarily paused updating the history on each PR.

v0.6.0
------

Major changes:

- Use `cftime.datetime` objects to represent datetimes instead
of `datetime.datetime` objects.  This results in times stored in a format compatible with
the fortran model, and accurate internal representation of times with the calendar specified
in the `coupler_nml` namelist.
- `Timer` class is added, with methods `start` and `stop`, and properties `clock` (context manager), and `times` (dictionary of accumulated timing)
- `CubedSphereCommunicator` instances now have a `.timer` attribute, which accumulates times for "pack", "unpack", "Isend", and "Recv" during halo updates
- make `SubtileGridSizer.from_tile_params` public API
- New method `CubedSphereCommunicator.synchronize_vector_interfaces` which synchronizes edge values on interface variables which are duplicated between adjacent ranks
- Added `.sel` method to corner views (e.g. `quantity.view.northeast.sel(x=0, y=1)`) to allow indexing these corner views with arbitrary dimension ordering.
- Halo updates now use tagged send/recv operations, which prevents deadlocks in certain situations
- Quantity.data is now guaranteed to be a numpy or cupy array matching its `.np` module, and will no longer be a gt4py Storage
- Quantity accepts a `gt4py_backend` on initialize which is used to create its `.storage` if one was not used on initialize
- parent MPI rank now referred to as "root" rank in variable names and documentation
- Added TILE_DIM constant for tile dimension of global quantities
- Added Partitioner base class implementing features necessary for scatter/gather
- Moved scatter and gather from TileCommunicator to the Communicator base class, so its code can be re-used by the CubedSphereCommunicator
- Implemented subtile_slice, global_extent, and subtile_extent routines on CubedSpherePartitioner necessary for scatter/gather in CubedSphereCommunicator
- Renamed argument `tile_extent` and `tile_dims` to `global_extent` and `global_dims` in routines to refer generically to the tile in the case of tile scatter/gather or cube in the case of cube scatter/gather
- Fixed a bug where initializing a Quantity with a numpy array and a gpu backend would give CPUStorage
- raise TypeError if initializing a quantity with both a storage and a gt4py_backend argument
- eagerly create storage object when initializing Quantity
- make data type of quantity and storage reflect the gt4py_backend chosen, instead of being determined based on the data type being numpy/cupy

Fixes:

- If `only_names` is provided to `open_restart`, it will return those fields and nothing more.  Previously it would include `"time"` in the returned state even if it was not requested.
- Fixed a bug where quantity.storage and quantity.data could be out of sync if the quantity was initialized using data and a gt4py backend string
- Default slice for corner views when not given at all as an index (e.g. when providing one index to a 2D view) now gives the same result as providing an empty slice (:)
- Fixed a bug where quantity.view could refer to a different array than quantity.data if the quantity was initialized using data and a gt4py backend string, and then quantity.storage was accessed

v0.5.1
------

- enable MPI tests on CircleCI

v0.5.0
------

Breaking changes:

- `send_buffer` and `recv_buffer` are modified to take in a `callable`, which is more easily serialized than a `numpy`-like module (necessary because we serialize the arguments to re-use buffers), and allows custom specification of the initialization if zeros are needed instead of empty.

Major changes:

- Added additional regional views to Quantity as attributes on Quantity.view, including `northeast`, `northwest`, `southeast`, `southwest`, and `interior`
- Separated fv3util into its own repository and began tracking history separately from fv3gfs-python
- Added getters and setters for additional dynamics quantities needed to call an alternative dynamical core
- Added `storage` property to Quantity, implemented as short-term shortcut to .data until gt4py GDP-3 is implemented

Deprecations:

- `Quantity.values` is deprecated

v0.4.3 (2020-05-15)
-------------------

Last release of fv3util with history contained in fv3gfs-python.
