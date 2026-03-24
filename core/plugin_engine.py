from pathlib import Path
from typing import Dict, Optional
import streamlit as st

from core.plugin_interface import PluginConfig
from core.plugin_parser import parse_plugin_file


class PluginEngine:
    """
    Engine for discovering and loading @tag-based plugins from /models.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self._plugins_cache: Dict[str, PluginConfig] = {}

    def discover_plugins(self) -> Dict[str, PluginConfig]:
        """Scans /models folder and returns parsed PluginConfig for all valid plugins."""
        plugins = {}

        for py_file in sorted(self.models_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            try:
                config = parse_plugin_file(py_file)
                if config:
                    plugin_id = py_file.stem
                    plugins[plugin_id] = config
                    self._plugins_cache[plugin_id] = config
            except Exception as e:
                st.warning(f"⚠️ Error loading plugin {py_file.name}: {e}")

        return plugins

    def get_plugin(self, plugin_id: str) -> Optional[PluginConfig]:
        """Returns a PluginConfig by its ID."""
        return self._plugins_cache.get(plugin_id)


@st.cache_resource
def get_plugin_engine() -> PluginEngine:
    """Returns a singleton plugin engine (cached by Streamlit)."""
    engine = PluginEngine()
    engine.discover_plugins()
    return engine
