# coding=utf-8
r"""
Sample file to generate visualizations.

- test gin config
`$ python -m bubble.main --agent=hello --gin_files=bubble/hello-bubble.gin`

- simple run
`$ python -m bubble.main --level=1 --steps=500 --restore_checkpoint=/tmp/bubble_dqn4/checkpoints/tf_ckpt-86`

- make viz video by agent
$ python -m bubble.main --agent=bubble --level=1 --steps=500 --restore_checkpoint=/tmp/bubble_dqn7/checkpoints/tf_ckpt-143

- more paremeters.
```
python -m bubble.main \
        --agent='bubble' \
        --game='BubbleBobble-Nes' \
        --steps=1000 \
        --root_dir='/tmp/bubble' \
        --restore_checkpoint=/tmp/checkpoints/colab_samples_rainbow_SpaceInvaders_v4_checkpoints_tf_ckpt-199
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
from absl import flags
from . import bubble_agent

flags.DEFINE_string('agent', 'bubble', 'Agent to visualize.')
flags.DEFINE_string('game', 'BubbleBobble', 'Game to visualize.')
flags.DEFINE_integer('level', 1, 'Game Level to Start')
flags.DEFINE_integer('steps', 200, 'Number of steps to run.')

flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', ["DQNAgent.tf_device = '/cpu:*'"],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')

flags.DEFINE_string('root_dir', '/tmp/bubble/', 'Root directory.')
flags.DEFINE_string('restore_checkpoint', None,
                    'Path to checkpoint to restore for visualizing.')
flags.DEFINE_boolean(
    'use_legacy_checkpoint', False,
    'Set to true if loading from a legacy (pre-Keras) checkpoint.')
FLAGS = flags.FLAGS

# run main
def main(_, hello='Hello'):
    # flag to use default config in bubble_agent.run.
    run_config = None

    # load gin configuration.
    if FLAGS.gin_files:
        from dopamine.discrete_domains import run_experiment
        print('! load gin-files:{}'.format(FLAGS.gin_files))
        run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
        run_config = ''         # do NOT load the default config in `bubble_agent.run`

    # run main.
    bubble_agent.run(agent=FLAGS.agent,
                     game=FLAGS.game,
                     level=FLAGS.level,
                     num_steps=FLAGS.steps,
                     root_dir=FLAGS.root_dir,
                     restore_ckpt=FLAGS.restore_checkpoint,
                     use_legacy_checkpoint=FLAGS.use_legacy_checkpoint,
                     config=run_config)

# main
if __name__ == '__main__':
    app.run(main)
