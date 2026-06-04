# Re-export desde 05_tools.py para compatibilidad con 06_agent.py
# (Python no puede importar módulos que empiezan por dígito directamente)
import importlib.util, sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_tools05", Path(__file__).parent / "05_tools.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TOOL_DEFINITIONS = _mod.TOOL_DEFINITIONS
dispatch_tool    = _mod.dispatch_tool
TOOLS_MAP        = _mod.TOOLS_MAP
