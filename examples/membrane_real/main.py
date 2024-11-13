# DETERMINISTIC
import os

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags


FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", ".", "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def set_cuda_visible_devices():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.config.gpu_id)

def main(argv):
    set_cuda_visible_devices()
    
    from examples.membrane_real.train import train_and_evaluate
    import examples.membrane_real.train as train
    import eval

    if FLAGS.config.mode == "train":
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval":
        eval.evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
