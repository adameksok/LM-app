import os
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List

MODELS_DIR = Path(__file__).parent.parent / "data" / "saved_models"

def _ensure_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def save_model(
    name: str,
    plugin_id: str,
    task: str,
    model_instance: Any,
    feature_names: List[str],
    target_name: str,
    user_params: Dict[str, Any]
) -> str:
    """Saves a model and its metadata to disk. Returns the unique model ID."""
    _ensure_dir()
    
    model_id = f"{plugin_id}_{int(time.time())}"
    filepath = MODELS_DIR / f"{model_id}.pkl"
    
    data = {
        "id": model_id,
        "name": name,
        "plugin_id": plugin_id,
        "task": task,
        "model": model_instance,
        "feature_names": feature_names,
        "target_name": target_name,
        "params": user_params,
        "created_at": time.time()
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
        
    return model_id

def load_model(model_id: str) -> Dict[str, Any]:
    """Loads a model dictionary by ID."""
    filepath = MODELS_DIR / f"{model_id}.pkl"
    if not filepath.exists():
        raise FileNotFoundError(f"Model {model_id} not found.")
        
    with open(filepath, "rb") as f:
        return pickle.load(f)

def list_saved_models() -> List[Dict[str, Any]]:
    """Returns a list of all saved models' metadata."""
    _ensure_dir()
    
    models = []
    for filepath in sorted(MODELS_DIR.glob("*.pkl"), key=os.path.getmtime, reverse=True):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                # Don't load the actual heavy model instance into memory just for listing
                models.append({
                    "id": data["id"],
                    "name": data["name"],
                    "plugin_id": data["plugin_id"],
                    "task": data["task"],
                    "feature_names": data["feature_names"],
                    "target_name": data["target_name"],
                    "created_at": data["created_at"]
                })
        except Exception:
            pass
            
    return models

def delete_model(model_id: str):
    """Deletes a saved model from disk."""
    filepath = MODELS_DIR / f"{model_id}.pkl"
    if filepath.exists():
        filepath.unlink()
