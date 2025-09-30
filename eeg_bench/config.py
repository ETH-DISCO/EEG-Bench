import json
import os
from pathlib import Path

_config = None

def load_config():
    global _config
    if _config is None:
        config_path = os.environ.get("EEG_BENCH_CONFIG")
        if config_path:
            config_path = Path(config_path).expanduser()
        else:
            script_dir = Path(__file__).resolve().parent
            config_path = script_dir / "config.json"
        
        if config_path.exists():
            with open(config_path, "r") as f:
                _config = json.load(f)
        else:
            _config = {}
    return _config

def get_config_value(key, default=None):
    # Check environment variable override
    env_key = f"EEG_BENCHMARK_{key.upper()}"
    if env_key in os.environ:
        return os.environ[env_key]

    config = load_config()
    return config.get(key, default)

def get_data_path(dataset_key=None, fallback_subdir=None):
    """
    Get path for a specific dataset or general data dir.
    Priority:
        1. Env var (e.g., EEG_BENCHMARK_TUEP)
        2. config.json value
        3. Default: <project_root>/data/<fallback_subdir>
    """
    if dataset_key:
        path = get_config_value(dataset_key)
        if path:
            return Path(path).expanduser()

    # Otherwise, use general "data" path from config or default
    base_data_path = get_config_value("data")
    if base_data_path:
        base_data_path = Path(base_data_path).expanduser()
    else:
        # Use project root's /data/ folder
        script_dir = Path(__file__).resolve().parent
        base_data_path = script_dir.parent / "data"

    if fallback_subdir:
        return base_data_path / fallback_subdir
    return base_data_path 
