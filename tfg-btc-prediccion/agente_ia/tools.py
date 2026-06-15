# Re-export desde 05_tools.py para compatibilidad con 06_agent.py
# (Python no puede importar módulos que empiezan por dígito directamente)
import importlib.util, sys
from pathlib import Path

# Instancia única: si el dashboard (u otro) ya cargó 05_tools bajo "tools05",
# reutilizarla. Dos instancias separadas tendrían cachés de precio distintas
# y el agente no vería el precio sembrado por el dashboard.
_mod = sys.modules.get("tools05")
if _mod is None:
    _spec = importlib.util.spec_from_file_location("tools05", Path(__file__).parent / "05_tools.py")
    _mod  = importlib.util.module_from_spec(_spec)
    sys.modules["tools05"] = _mod
    _spec.loader.exec_module(_mod)

TOOL_DEFINITIONS = _mod.TOOL_DEFINITIONS
dispatch_tool    = _mod.dispatch_tool
TOOLS_MAP        = _mod.TOOLS_MAP
