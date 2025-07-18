import json

def load_json(file):
    try:
        return json.load(file)
    except Exception:
        return None
