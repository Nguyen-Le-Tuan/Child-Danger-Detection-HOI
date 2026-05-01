import json, datetime
from os import path

def write_report(obj, path):
    with open(path, 'w') as f:
        json.dump({'time': datetime.datetime.utcnow().isoformat(), 'report': obj}, f, indent=2)