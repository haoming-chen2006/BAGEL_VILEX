from omegaconf import Omegaconf

def load_config(path):
    return Omegaconf.load(path)