import os
import numpy as np
import tensorflow as tf
from utils.common_utils import get_logger, delete_directory


def class_weight(labels):
    """
    Calculate class weight (following the formula mentioned in sklearn.utils
    class_weight.compute_class_weight)

    Args:
        labels: [N,] numpy array, int labels of train dataset.
    Return:
        weights: [C,] numpy array, float weights of C classes.
    """
    bincount = np.bincount(labels)
    num_samples = len(labels)
    num_classes = len(bincount)
    weights = num_samples / (num_classes * bincount)
    return weights


class ModelSitter:
    """
    A baby-sitter for training and testing model, support logger, tf_summary,
    automatically checkpoint, etc.
    """
    def __init__(self, log_dir, model, optimizer=None, mode="train", ckpt_period=1, max_ckpts=1):
        if mode not in ["train", "refine", "eval"]:
            raise ValueError(f"expect mode of [\"train\", \"refine\", \"eval\"], got {mode}")

        # necessaries
        self.model = model
        self.optimizer = optimizer
        with tf.device("CPU:0"):
            self._overall_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="overall_step")
            self._best_critical = tf.Variable(-np.inf, trainable=False, dtype=tf.double, name="best_critical")
        self.mode = mode

        # logger
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.logger = get_logger(mode, loglevel="debug", logfile=os.path.join(self.log_dir, f"{mode}_log.txt"))

        # TF summary
        self.summary_dir = os.path.join(self.log_dir, "tf_summary")
        if self.mode != "eval" and os.path.exists(self.summary_dir): delete_directory(self.summary_dir)  # remove old summary
        self._summary_writer_dict = {}
        self._summary_step_dict = {}

        # checkpoints
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        self.best_weight_prefix = os.path.join(self.log_dir, "best_weight", "best")
        self.ckpt_period = ckpt_period
        self.max_ckpts = max_ckpts
        if optimizer is None:
            self._ckpt = tf.train.Checkpoint(model=model, steps=self._overall_step, critical=self._best_critical)
        else:
            self._ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, steps=self._overall_step, critical=self._best_critical)
        self._ckpt_manager = tf.train.CheckpointManager(self._ckpt, self.ckpt_dir, self.max_ckpts, checkpoint_name="ckpt")

    @property
    def overall_step(self):
        return self._overall_step.numpy()

    @property
    def best_critical(self):
        return self._best_critical.numpy()

    @property
    def checkpoints(self):
        return self._ckpt_manager.checkpoints


    # ----------------- #
    # model checkpoints #
    # ----------------- #

    def step(self, critical=None):
        self._overall_step.assign_add(1)

        # update checkpoint
        if self._overall_step % self.ckpt_period == 0:
            self._ckpt_manager.save(self._overall_step.numpy())

        # update best critical
        if critical is not None and critical > self.best_critical:
            self._best_critical.assign(critical)
            self.model.save_weights(self.best_weight_prefix)
            self.info_log("Best critical improved, current: {:.4f}, steps:{:d}".format(critical, self._overall_step.numpy()))

    def restore_ckpt(self, ckpt=None, index=None):
        if ckpt is not None:
            if index is not None: print("passed ckpt and index simultaneously, ignoring index.")
        else:
            ckpt = self._ckpt_manager.checkpoints[index]
        status = self._ckpt.restore(ckpt)
        #! ideally, when training or refining, we should assert all saved stuff are consumed
        #! properly (as the following annotated code), however, tf optimizers like Adam keep
        #! a variable named iter for trainable variables optimized by self, which will never
        #! exist after the optimizer is created until the optimizer is actually called, this
        #! lead to the inconsistend between the checkpoint and newly created optimizer, thus,
        #! assert_consumed will always failed and assertion error like "Unresolved object in
        #! checkpoint (root).optimizer.iter" will always raised. This issue is not solved in
        #! tf 2.3.1, so we temporarily use assert_existing_objects_matched at all situation.
        #! ref: https://github.com/tensorflow/tensorflow/issues/33150
        # if self.mode == "eval":
        #     status.assert_existing_objects_matched()
        # else:
        #     status.assert_consumed()
        status.assert_existing_objects_matched()

    def restore_latest_ckpt(self):
        if not self._ckpt_manager.latest_checkpoint:
            print(f"Cannot find latest checkpoint in {self.ckpt_dir}!")
            return
        status = self._ckpt.restore(self._ckpt_manager.latest_checkpoint)
        # if self.mode == "eval":
        #     status.assert_existing_objects_matched()
        # else:
        #     status.assert_consumed()
        status.assert_existing_objects_matched()

    def load_best_weight(self):
        if not self.model.built:
            print("You need to build model first before loading weights!")
        elif not os.path.isdir(os.path.dirname(self.best_weight_prefix)):
            print("Best weight not found!")
        self.model.load_weights(self.best_weight_prefix)


    # ---------------------- #
    #   tensorboard summary  #
    # ---------------------- #

    def scalar_summary(self, name, data, writer=None, step=None, reuse_last_step=False):
        writer = self._get_summary_writer(writer)
        step = self._get_summary_step(name, step, reuse=reuse_last_step)

        with writer.as_default():
            tf.summary.scalar(name, data, step)

    def reset_summary(self):
        # remove summary dir
        if os.path.exists(self.summary_dir):
            os.rmdir(self.summary_dir)
        # reset writers and steps
        for writer in self._summary_writer_dict:
            self._summary_writer_dict[writer].close()
        self._summary_writer_dict.clear()
        self._summary_step_dict.clear()

    def _get_summary_writer(self, writer):
        writer = "." if writer is None else writer
        if writer not in self._summary_writer_dict:
            self._summary_writer_dict[writer] = tf.summary.create_file_writer(self.summary_dir + "/" + writer)
        return self._summary_writer_dict[writer]

    def _get_summary_step(self, name, step, reuse=False):
        if step is None:
            if name not in self._summary_step_dict:
                self._summary_step_dict[name] = 1
            elif not reuse:
                self._summary_step_dict[name] += 1
            step = self._summary_step_dict[name]
        return step


    # ------------------ #
    #  logger shortcuts  #
    # ------------------ #

    def debug_log(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info_log(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning_log(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error_log(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical_log(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
