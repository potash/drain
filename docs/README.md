## Definitions

### High level
Step: the basic component of drain. Steps inherit from the `drain.step.Step` baseclass and implement a `run()` method. A step is a deterministic function from its consturctor arguments to its result.

Result: The return value of a step's `run()`.

Input: A step is an input to another step if the first depends on the second. A collection of steps on which a given step depends.

Graph (or workflow): A collection of steps with (directed) edges representing input relationships.
Root: A step which has no input steps.
Leaf: The final step in a workflow.
Branch: The subgraph consisisting of a leaf and all of its dependencies.

### Execution
Target segment: a path (or a collection of paths) in a workflow which ends with a target and starts with either a root 

#### directory structure
{drain.PATH}/{self._digest}/
 - step.yaml
 - dump/
 - target
     
