"""Top-level configuration for the Riemann quantum graph search.

Re-exports SearchConfig from the orchestrator module so that callers can do
``from config import SearchConfig`` without reaching into the agents package.
"""

from riemann_qg.agents.orchestrator import SearchConfig

__all__ = ["SearchConfig"]
