import yaml
import os

from drain import util
from drain.step import Step


def load(filename):
    """
    Load step from yaml file
    Args:
        filename: a target or step.yaml filename
    """
    yaml_filename = os.path.join(os.path.dirname(filename), 'step.yaml')
    with open(yaml_filename) as f:
        return yaml.load(f)


def step_multi_representer(dumper, data):
    tag = '!step:%s.%s' % (data.__class__.__module__, data.__class__.__name__)

    return dumper.represent_mapping(tag, data.get_arguments())


def step_multi_constructor(loader, tag_suffix, node):
    cls = util.get_attr(tag_suffix[1:])
    kwargs = loader.construct_mapping(node, deep=True)

    return cls(**kwargs)


def configure():
    """
    Configures YAML parser for Step serialization and deserialization
    Called in drain/__init__.py
    """
    yaml.add_multi_representer(Step, step_multi_representer)
    yaml.add_multi_constructor('!step', step_multi_constructor)
    yaml.Dumper.ignore_aliases = lambda *args: True
