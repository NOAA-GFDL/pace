# flake8: noqa

# This magical series of imports is to de-duplicate the conftest.py file
# between the dycore and physics tests. We can avoid this if we refactor the tests
# to all run from one directory

import pyFV3.testing


# this must happen before any classes from pyFV3 are instantiated
pyFV3.testing.enable_selective_validation()

import ndsl.stencils.testing.conftest
from ndsl.stencils.testing.conftest import *  # noqa: F403,F401

from . import translate


ndsl.stencils.testing.conftest.translate = translate  # type: ignore
