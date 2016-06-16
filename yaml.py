from __future__ import absolute_import
import yaml

from cached_property import cached_property
from drain import util
from drain.step import Step

# load step(s) via templates
def load(filename):
    with open(filename) as f:
        templates = yaml.load(f)
        if isinstance(templates, StepTemplate):
            return templates.step
        elif hasattr(templates, '__iter__'):
            return [t.step for t in templates]

# temporary holder of step arguments
# useful to get around pyyaml bug: https://bitbucket.org/xi/pyyaml/issues/56/sub-dictionary-unavailable-in-constructor
class StepTemplate(object):
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
    return dumper.represent_mapping(tag, data.get_arguments(
            inputs=True, inputs_mapping=True, dependencies=True))

def step_multi_constructor(loader, tag_suffix, node):
    cls = util.get_attr(tag_suffix[1:])
    kwargs = loader.construct_mapping(node)

    return StepTemplate(_cls=cls, **kwargs)

def configure():
    yaml.add_multi_representer(Step, step_multi_representer)
    yaml.add_multi_constructor('!step', step_multi_constructor)
 
