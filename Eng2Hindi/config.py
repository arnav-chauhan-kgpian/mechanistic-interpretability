from pathlib import Path

def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 20,
        "lr": 2e-4,
        "seq_len": 128,
        "d_model": 512, # Changed from 360 to 512 (Standard practice: 512/8 = 64 per head)
        "lang_src": "en",
        "lang_tgt": "hi", # CHANGED: Tamil (ta) -> Hindi (hi)
        "model_folder": "weights",
        "model_basename": "tmodel_", # CHANGED: Just the prefix, not the full path
        "preload": None, # Set to 'latest' to resume training if interrupted
        "tokenizer_file": "tokenizer_{0}.json",          
        "experiment_name": "runs/tmodel",
        "N": 6,
        "h": 8,
        "dropout": 0.1 # Reduced slightly (0.2 is okay, but 0.1 is standard for this size)
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    # CHANGED: Removed dependency on missing 'datasource' key
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    
    # Check if folder exists
    if not Path(model_folder).exists():
        return None
        
    model_filename = f"{model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    
    if len(weights_files) == 0:
        return None
    
    weights_files.sort()
    return str(weights_files[-1])