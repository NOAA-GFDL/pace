import warnings
from timeit import default_timer as time
from typing import Mapping

from ndsl.optional_imports import cupy as cp
from ndsl.utils import GPU_AVAILABLE


class Timer:
    """Class to accumulate timings for named operations."""

    def __init__(self):
        self._clock_starts = {}
        self._accumulated_time = {}
        self._hit_count = {}
        self._enabled = True
        # Check if we have CUDA device and it's ready to
        # perform tasks
        self._can_time_CUDA = GPU_AVAILABLE

    def start(self, name: str):
        """Start timing a given named operation."""
        if self._can_time_CUDA:
            cp.cuda.Device(0).synchronize()
            cp.cuda.nvtx.RangePush(name)
        if self._enabled:
            if name in self._clock_starts:
                raise ValueError(f"clock already started for '{name}'")
            else:
                self._clock_starts[name] = time()

    def stop(self, name: str):
        """Stop timing a given named operation, add the time elapsed to
        accumulated timing and increase the hit count.
        """
        if self._can_time_CUDA:
            cp.cuda.Device(0).synchronize()
            cp.cuda.nvtx.RangePop()
        if self._enabled:
            if name not in self._accumulated_time:
                self._accumulated_time[name] = time() - self._clock_starts.pop(name)
            else:
                self._accumulated_time[name] += time() - self._clock_starts.pop(name)
            if name not in self._hit_count:
                self._hit_count[name] = 1
            else:
                self._hit_count[name] += 1

    def clock(self, name: str):
        """Context manager to produce timings of operations.

        Args:
            name: the name of the operation being timed

        Example:
            The context manager times operations that happen within its context. The
            following would time a time.sleep operation::

                >>> import time
                >>> from ndsl.performance.timer import Timer
                >>> timer = Timer()
                >>> with timer.clock("sleep"):
                ...     time.sleep(1)
                ...
                >>> timer.times
                {'sleep': 1.0032463260000029}
        """

        # [DaCe] Because the contextlib is a "one-shot" object
        # which self-destroys itself when called, we can't orchestrate
        # it easily in DaCe. Waiting for a fix DaCe side to this Python
        # ridiculousness (see contelib.py:_GeneratorContextManager.__enter__)
        def dace_inhibitor(func):
            return func

        class Wrapper:
            def __init__(self, timer, name) -> None:
                self.timer = timer
                self.name = name

            @dace_inhibitor
            def __enter__(self):
                self.timer.start(name)

            @dace_inhibitor
            def __exit__(self, type, value, traceback):
                self.timer.stop(name)

        return Wrapper(self, name)

    @property
    def times(self) -> Mapping[str, float]:
        """accumulated timings for each operation name"""
        if len(self._clock_starts) > 0:
            warnings.warn(
                "Retrieved times while clocks are still going, "
                "incomplete times are not included: "
                f"{list(self._clock_starts.keys())}",
                RuntimeWarning,
            )
        return self._accumulated_time.copy()

    @property
    def hits(self) -> Mapping[str, int]:
        """accumulated hit counts for each operation name"""
        if len(self._clock_starts) > 0:
            warnings.warn(
                "Retrieved hit counts while clocks are still going, "
                "incomplete times are not included: "
                f"{list(self._clock_starts.keys())}",
                RuntimeWarning,
            )
        return self._hit_count.copy()

    def reset(self):
        """Remove all accumulated timings."""
        self._accumulated_time.clear()
        self._hit_count.clear()

    def enable(self):
        """Enable the Timer."""
        self._enabled = True

    def disable(self):
        """Disable the Timer."""
        if len(self._clock_starts) > 0:
            raise RuntimeError(
                "Cannot disable timer while clocks are still going: "
                f"{list(self._clock_starts.keys())}"
            )
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Indicates whether the timer is currently enabled."""
        return self._enabled


class NullTimer(Timer):
    """A Timer class which does not actually accumulate timings.

    Meant to be used in place of an optional timer.
    """

    def __init__(self):
        super().__init__()
        self._enabled = False

    def enable(self):
        """Enable the Timer."""
        raise NotImplementedError(
            "NullTimer cannot be enabled, maybe create a Timer and "
            "disable it instead of using NullTimer"
        )

    @property
    def enabled(self) -> bool:
        """Indicates whether the timer is currently enabled."""
        return False
