"""Error taxonomy for IntentForge.

All errors derive from :class:`IntentForgeError` and carry a stable error code
in the form ``E<module><seq>`` so callers and the CLI can branch on them.

Code ranges
-----------
E100  configuration / environment
E200  data loading / generation
E300  backend (301 import, 302 fit, 303 predict)
E400  router (401 calibration, 402 routing)
E500  pipeline orchestration
"""

from __future__ import annotations


class IntentForgeError(Exception):
    """Base class for all IntentForge errors."""

    code = "E000"

    def __init__(self, message: str, *, code: str | None = None, cause: BaseException | None = None):
        self.code = code or self.code
        self.message = message
        self.cause = cause
        super().__init__(f"[{self.code}] {message}")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# --- E100 configuration -----------------------------------------------------
class ConfigError(IntentForgeError):
    code = "E100"


# --- E200 data --------------------------------------------------------------
class DataError(IntentForgeError):
    code = "E200"


# --- E300 backend -----------------------------------------------------------
class BackendError(IntentForgeError):
    code = "E300"


class BackendImportError(BackendError):
    code = "E301"


class BackendFitError(BackendError):
    code = "E302"


class BackendPredictError(BackendError):
    code = "E303"


# --- E400 router ------------------------------------------------------------
class RouterError(IntentForgeError):
    code = "E400"


class RouterCalibrationError(RouterError):
    code = "E401"


class RouterRouteError(RouterError):
    code = "E402"


# --- E500 pipeline ----------------------------------------------------------
class PipelineError(IntentForgeError):
    code = "E500"
