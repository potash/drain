from __future__ import absolute_import
import yaml
import os

from cached_property import cached_property
from drain import util
from drain.step import Step

def load(filename):
    """
    Load step from yaml file (via template)
    Args:
        filename: a target or step.yaml filename
    """
    yaml_filename = os.path.join(os.path.dirname(filename), 'step.yaml')
    with open(yaml_filename) as f:
        template = yaml.load(f)
        return template.step

class StepTemplate(object):
    """
    Temporary holder of step arguments
    Used to get around pyyaml bug: https://bitbucket.org/xi/pyyaml/issues/56/sub-dictionary-unavailable-in-constructor
    """
    def __init__(self, _cls, name=None, target=False, **kwargs):
        self._cls = _cls
        self.name = name
        self.target = target
        self.kwargs = kwargs

    # it's important that this is cached so that multiple calls to step return the same Step object
    @cached_property
    def step(self):
        if 'inputs' in self.kwargs:
            self.kwargs['inputs'] = [t.step for t in self.kwargs['inputs']]

        return self._cls(target=self.target, name=self.name, 
                **self.kwargs)

def step_multi_representer(dumper, data):
    tag = '!step:%s.%s' % (data.__class__.__module__, data.__class__.__name__)

    return dumper.represent_mapping(tag, data.get_arguments())

def step_multi_constructor(loader, tag_suffix, node):
    cls = util.get_attr(tag_suffix[1:])
    kwargs = loader.construct_mapping(node)

    return StepTemplate(_cls=cls, **kwargs)

def configure():
    """
    Configures YAML parser for Step serialization and deserialization
    Called in drain/__init__.py
    """
    yaml.add_multi_representer(Step, step_multi_representer)
    yaml.add_multi_constructor('!step', step_multi_constructor)
