# coding=utf-8
r"""
Sample file to generate visualizations.

- simple run
`$ python -m bubble.main --level=2`

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

flags.DEFINE_string('root_dir', '/tmp/bubble/', 'Root directory.')
flags.DEFINE_string('restore_checkpoint', None,
                    'Path to checkpoint to restore for visualizing.')
flags.DEFINE_boolean(
    'use_legacy_checkpoint', False,
    'Set to true if loading from a legacy (pre-Keras) checkpoint.')
FLAGS = flags.FLAGS

# run main


def main(_):
    # python -m bubble.main --agent=hello
    if FLAGS.agent == 'hello':
        print('hello %s' % (FLAGS.game))
        exit(-1)

    # run main
    bubble_agent.run(agent=FLAGS.agent,
                     game=FLAGS.game,
                     level=FLAGS.level,
                     num_steps=FLAGS.steps,
                     root_dir=FLAGS.root_dir,
                     restore_ckpt=FLAGS.restore_checkpoint,
                     use_legacy_checkpoint=FLAGS.use_legacy_checkpoint)


if __name__ == '__main__':
    app.run(main)
