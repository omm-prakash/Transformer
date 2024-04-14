from pathlib import Path
import time, datetime
import yaml

class Time():
    def __init__(self):
        self.begin = 0
        self.final = 0
    def now(self):
        return datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    def reset(self):
        self.begin = time.time()
        self.final = time.time()        
    def start(self, message=None):
        # if message:
        self.message = message
        self.begin = time.time()
    def end(self):
        self.final = time.time()
        tm = float(self.final-self.begin)
        unit = 'sec'
        if tm > 60:
            tm = tm/60
            unit = 'min'
        elif tm > 3600:
            tm = tm/3600
            unit = 'hr'
        if self.message:
            print('>> {}: Done!! Time taken: {:.4f} {}'.format(self.message, tm, unit))
        else:
            print('>> Done!! Time taken: {:.4f} {}'.format(tm, unit))
        self.message = None


def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
            return config_data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None

def get_weights_file_path(config, epoch: str):
        model_folder = f"{(config['dataset_config_name'])}-{config['model_folder']}"
        model_filename = f"{config['model_basename']}{epoch}.pt"
        return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
        model_folder = f"{(config['dataset_config_name'])}-{config['model_folder']}"
        model_filename = f"{config['model_basename']}*"
        weights_files = list(Path(model_folder).glob(model_filename))
        if len(weights_files) == 0:
            return None
        weights_files.sort()
        return str(weights_files[-1])