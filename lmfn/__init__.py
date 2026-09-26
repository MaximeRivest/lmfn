"""lmfn — the function is the prompt; calling it calls the model.

Built on lmcc (how each value is written and read) and lm15 (the wire).
"""

from .core import (CallResult, Function, StepLimit, ai, configure, default_adapter, json_adapter,
                   router, use_router)
from .session import Example, Session

__version__ = "0.1.0"
__all__ = ["CallResult", "Example", "Function", "Session", "StepLimit", "ai", "configure",
           "default_adapter", "json_adapter", "router", "use_router"]
