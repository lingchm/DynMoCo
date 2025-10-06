import numpy as np 

def convert_keys_to_builtin(obj):
    if isinstance(obj, dict):
        return {convert_keys_to_builtin_key(k): convert_keys_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_builtin(i) for i in obj]
    else:
        return obj

def convert_keys_to_builtin_key(key):
    if isinstance(key, (np.integer,)):
        return int(key)
    elif isinstance(key, (np.floating,)):
        return float(key)
    elif isinstance(key, bytes):
        return key.decode()
    else:
        return key
    
def convert_inner_keys_to_int(d):
    if isinstance(d, dict):
        return {
            k: {int(inner_k): v2 for inner_k, v2 in v.items()} if isinstance(v, dict) else v
            for k, v in d.items()
        }
    return d