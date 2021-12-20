# parse args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=str, default="0", help="Idx of GPU to use (split with ','). [default: 0]")
parser.add_argument("--dataset", type=str, default="scannet", help="Dataset name. scannet or sunrgbd. [default: scannet]")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size. [default: 8]")
parser.add_argument("--init_lr", type=float, default=0.0015, help="Initial learning rate. [default: 0.0015]")
parser.add_argument("--log_dir", type=str, default="logs", help="directory for saving logs and checkpoints. [default: logs]")
parser.add_argument("--verbose_interval", type=int, default=15, help="Verbose period over steps. [default: 15]")
parser.add_argument("--eval_period", type=int, default=5, help="Evaluate period over epochs. [default: 5]")
parser.add_argument("--ckpt_period", type=int, default=10, help="checkpoint period over epochs. [default: 10]")
parser.add_argument("--max_ckpt", type=int, default=1, help="Maximum number of checkpoint to keep. [default: 1]")
parser.add_argument("--keep_all_classes", action="store_true", help="Whether to keep all classes of class with maximum prob when parse proposals.")
args = parser.parse_args()


# config
config = {
    # basic config
    "dataset": args.dataset,  # ScanNet or SUNRGBD
    "downsample": 40000,
    "detect_proposals": 256,

    # loss config
    "near_threshold": 0.3,
    "far_threshold": 0.6,

    # train config
    "gpu": [int(idx) for idx in args.gpus.split(",")],
    "max_epochs": 180,
    "batch_size": args.batch_size,
    "prefetch": 6,
    "cache_data": False,

    "initial_lr": args.init_lr,
    "minimal_lr": 1E-7,
    "lr_decay_thres": [80, 120, 160],
    "lr_decay_rates": [0.1, 0.1, 0.1],

    "initial_bnm": 0.5,
    "minimal_bnm": 0.001,
    "bnm_decay_rate": 0.5,
    "bnm_decay_period": 20,

    # evaluate config
    "eval_period": args.eval_period,
    "nms_type": 1,  # 0: 2D, 1: 3D
    "inclass_nms": True,
    "nms_iou_threshold": 0.25,
    "objectness_threshold": 0.05,
    "ap_iou_threshold": 0.25,
    "keep_all_classes": args.keep_all_classes,
    "use_07_metric": False,
    "num_worker": 8,

    # log config
    "log_dir": args.log_dir,
    "ckpt_period": args.ckpt_period,
    "max_ckpt": args.max_ckpt,
    "verbose_interval": args.verbose_interval,
}


# headers
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model.network import DetectModel
from model.util import get_losses, parse_prediction, pack_prediction, pack_groundtruth, unpack_prediction, unpack_groundtruth, box3d_iou
from utils.metric_utils import APMetric
from utils.trainval_utils import ModelSitter


# Ensure TensorFlow only use specified GPUs
gpu_devices = tf.config.list_physical_devices("GPU")
assert len(gpu_devices) >= len(config["gpu"]), "Come on, man!"
tf.config.set_visible_devices([gpu_devices[i] for i in config["gpu"]], "GPU")
for i in config["gpu"]: tf.config.experimental.set_memory_growth(gpu_devices[i], True)


# get dataset utils
if config["dataset"].lower() == "scannet":
    from utils.dataset_utils.ScanNet.scannet_utils import META
    from utils.dataset_utils.ScanNet.dataloader import get_dataset
    config["dataset_dir"] = "datasets/ScanNet"
    config["max_instances"] = META.MAX_BOXES
    config["detect_classes"] = len(META.detect_target_nyuid)
    config["class_weight"] = np.array(  #! UNUSED
        [0.61251265, 2.8470865 , 0.20060949, 2.1528462 , 0.68769123,
         0.43141933, 0.94187021, 2.91351852, 1.32232308, 4.0465535 ,
         1.58630772, 2.99334094, 4.69922342, 7.53496169, 4.3485351 ,
         2.24116809, 7.73500492, 0.44033025],
        dtype=np.float32
    )
    config["num_heading_bin"] = 1
    config["mean_size"] = np.array(
        [[0.7750491 , 0.94897728, 0.96542059],
         [1.86903267, 1.83214713, 1.19222991],
         [0.61214779, 0.6192873 , 0.70480848],
         [1.44113897, 1.60452037, 0.83652295],
         [1.04780726, 1.20164188, 0.63457007],
         [0.56101231, 0.60847217, 1.71950401],
         [1.07894896, 0.82033996, 1.16921199],
         [0.84171093, 1.35047944, 1.6898925 ],
         [0.23051738, 0.47640499, 0.56569256],
         [1.45484898, 1.97119893, 0.28643281],
         [1.07858031, 1.53705114, 0.86501906],
         [1.43119644, 0.76923112, 1.64982672],
         [0.62969195, 0.70871286, 1.31433587],
         [0.43925034, 0.41569593, 1.70002748],
         [0.58504462, 0.57878438, 0.72029612],
         [0.51158692, 0.50960674, 0.3128736 ],
         [1.1732076 , 1.05987141, 0.51812528],
         [0.57052652, 0.60265939, 0.64827649]],
        dtype=np.float32
    )
    config["parse_heading"] = False

elif config["dataset"].lower() == "sunrgbd":
    from utils.dataset_utils.SUNRGBD.sunrgbd_utils import META
    from utils.dataset_utils.SUNRGBD.dataloader import get_dataset
    config["dataset_dir"] = "datasets/SUNRGBD"
    config["max_instances"] = META.MAX_BOXES
    config["detect_classes"] = len(META.detect_target_nyuid)
    config["class_weight"] = np.array(  #! UNUSED
        [1.97157233, 0.58682142, 2.16491713, 0.16500684,  8.85536723,
         1.52470817, 8.12124352, 5.19006623, 7.39339623, 21.47123288],
        dtype=np.float32
    )
    config["num_heading_bin"] = 12
    config["mean_size"] = np.array(
        [[2.00852079, 1.53821184, 1.02429467],
         [0.88851368, 1.43826426, 0.71595464],
         [0.99135877, 1.82947722, 0.8422245 ],
         [0.58785727, 0.5555384 , 0.84075032],
         [0.69277293, 0.46631492, 0.73081437],
         [0.71902821, 1.40403822, 0.78850871],
         [0.56894714, 1.06392299, 1.17279978],
         [0.50873764, 0.63245229, 0.6990666 ],
         [-1.44911839, 1.55167933, 1.55878774],
         [0.80223968, 1.38127347, 0.48278037]],
        dtype=np.float32
    )
    config["parse_heading"] = True
else:
    raise ValueError("Unknown dataset, expect \"ScanNet\" or \"SUNRGBD\", got {}".format(config["dataset"]))

id_to_name = { k:META.nyuid_to_class[META.id_to_label[k]] for k in META.id_to_label.keys() }


# generate datasets
#! WARNING: if training with distribute strategy (i.e. multi-GPU), I insist to DROP THE
#! REMAINDER DATA when batching, as if the remaining data cannot split to all devices (
#! i.e. total_batches % batch_size < num_devices, some device has no data to run with),
#! a warning of "WARNING:tensorflow:Gradients do not exist for variables" will raised,
#! which will further cause AssertionError in "distribute_utils.py" as it assert model
#! on all device has the same shape of gradient, which is impossible for idled devices.
train_dataset = get_dataset(data_dir=config["dataset_dir"],
                            max_instances=config["max_instances"],
                            shuffle=True,
                            augment=True,
                            downsample=config["downsample"],
                            split="train",
                            cache_data=config["cache_data"])
train_dataset = train_dataset.batch(config["batch_size"], True).prefetch(config["prefetch"])

test_dataset = get_dataset(data_dir=config["dataset_dir"],
                            max_instances=config["max_instances"],
                            shuffle=True,
                            augment=False,
                            downsample=config["downsample"],
                            split="val",
                            cache_data=config["cache_data"])
test_dataset = test_dataset.batch(config["batch_size"], False).prefetch(config["prefetch"])


# determine strategy
if len(config["gpu"]) == 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:{:d}".format(config["gpu"][0]))
elif len(config["gpu"]) > 1:
    strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{idx}" for idx in config["gpu"]])
else:
    raise NotImplementedError("Man, you don't want to run this on CPU, trust me!")


# generate training basics
with strategy.scope():
    model = DetectModel(num_class=config["detect_classes"],
                        num_proposal=config["detect_proposals"],
                        num_heading_bin=config["num_heading_bin"],
                        mean_size=config["mean_size"])
    optimizer = keras.optimizers.Adam(learning_rate=config["initial_lr"], beta_1=0.9, beta_2=0.99)

    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    sitter = ModelSitter(config["log_dir"], model=model, optimizer=optimizer,
                         mode="train", ckpt_period=config["ckpt_period"], max_ckpts=config["max_ckpt"])

ap_metric = APMetric(box3d_iou, pred_unpack_func=unpack_prediction, gt_unpack_func=unpack_groundtruth)
total_loss_metric = keras.metrics.Mean(name="mean_total_loss", dtype=np.float32)
vote_loss_metric = keras.metrics.Mean(name="mean_vote_loss", dtype=np.float32)
obj_loss_metric = keras.metrics.Mean(name="mean_objectness_loss", dtype=np.float32)
box_loss_metric = keras.metrics.Mean(name="mean_box_loss", dtype=np.float32)


# train & val functions
def reset_loss_metrics():
    total_loss_metric.reset_states()
    vote_loss_metric.reset_states()
    obj_loss_metric.reset_states()
    box_loss_metric.reset_states()

@tf.function
def train_one_step(data_dict):

    with tf.GradientTape() as tape:
        pred_dict = model(inputs=data_dict["points"], training=True)

        losses, vote_loss, objectness_loss, box_loss = get_losses(
            data_dict, pred_dict,
            near_threshold=config["near_threshold"],
            far_threshold=config["far_threshold"],
            num_heading_bin=config["num_heading_bin"],
            num_class=config["detect_classes"],
            mean_size=config["mean_size"]
        )
        losses = losses / config["batch_size"]
    vote_loss = vote_loss / config["batch_size"]
    objectness_loss = objectness_loss / config["batch_size"]
    box_loss = box_loss / config["batch_size"]

    grads = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return losses, vote_loss, objectness_loss, box_loss

@tf.function
def evaluate_one_step(data_dict):
    pred_dict = model(inputs=data_dict["points"], training=False)
    objectness, center, size, heading, label = parse_prediction(
        pred_dict,
        num_heading_bin=config["num_heading_bin"],
        mean_size=config["mean_size"],
        parse_heading=config["parse_heading"]
    )
    class_probs = tf.nn.softmax(pred_dict["scores"], axis=2)

    losses, vote_loss, objectness_loss, box_loss = get_losses(
        data_dict, pred_dict,
        near_threshold=config["near_threshold"],
        far_threshold=config["far_threshold"],
        num_heading_bin=config["num_heading_bin"],
        num_class=config["detect_classes"],
        mean_size=config["mean_size"]
    )

    losses = losses / config["batch_size"]
    vote_loss = vote_loss / config["batch_size"]
    objectness_loss = objectness_loss / config["batch_size"]
    box_loss = box_loss / config["batch_size"]

    return \
        objectness, center, size, heading, label, class_probs, \
        losses, vote_loss, objectness_loss, box_loss

def train_one_epoch():
    interval = config["verbose_interval"]
    reset_loss_metrics()

    for step, data_dict in enumerate(train_dataset):
        # run one training step with strategy
        losses, vote_loss, objectness_loss, box_loss = strategy.run(train_one_step, args=(data_dict,))

        # aggregate values from all devices
        total_loss_metric.update_state(strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None))
        vote_loss_metric.update_state(strategy.reduce(tf.distribute.ReduceOp.SUM, vote_loss, axis=None))
        obj_loss_metric.update_state(strategy.reduce(tf.distribute.ReduceOp.SUM, objectness_loss, axis=None))
        box_loss_metric.update_state(strategy.reduce(tf.distribute.ReduceOp.SUM, box_loss, axis=None))

        # loss log (summary step is mini-batch num)
        if (step + 1) % interval == 0:
            losses = total_loss_metric.result().numpy()
            vote_loss = vote_loss_metric.result().numpy()
            objectness_loss = obj_loss_metric.result().numpy()
            box_loss = box_loss_metric.result().numpy()
            reset_loss_metrics()

            sitter.scalar_summary("losses/vote_loss", vote_loss, writer="train")
            sitter.scalar_summary("losses/objectness_loss", objectness_loss, writer="train")
            sitter.scalar_summary("losses/box_loss", box_loss, writer="train")

            sitter.logger.info(
                "STEP {:<4d} - total: {:.5f}, vote: {:.5f}, objectness: {:.5f}, box: {:.5f}".format(
                step + 1, losses, vote_loss, objectness_loss, box_loss)
            )

def evaluate_one_epoch():
    reset_loss_metrics()

    # predict
    for data_dict in test_dataset:
        # run one evaluating step with strategy
        objectness, center, size, heading, label, class_probs, \
        losses, vote_loss, objectness_loss, box_loss = strategy.run(evaluate_one_step, args=(data_dict,))

        # aggregate values from all device (for latest TF, one can just use strategy.gather)
        for key in data_dict:
            data_dict[key] = tf.concat(strategy.experimental_local_results(data_dict[key]), axis=0)
        center = tf.concat(strategy.experimental_local_results(center), axis=0)
        size = tf.concat(strategy.experimental_local_results(size), axis=0)
        heading = tf.concat(strategy.experimental_local_results(heading), axis=0)
        objectness = tf.concat(strategy.experimental_local_results(objectness), axis=0)
        label = tf.concat(strategy.experimental_local_results(label), axis=0)
        class_probs = tf.concat(strategy.experimental_local_results(class_probs), axis=0)

        # update loss metric
        total_loss_metric.update_state(strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None))
        vote_loss_metric.update_state(strategy.reduce(tf.distribute.ReduceOp.SUM, vote_loss, axis=None))
        obj_loss_metric.update_state(strategy.reduce(tf.distribute.ReduceOp.SUM, objectness_loss, axis=None))
        box_loss_metric.update_state(strategy.reduce(tf.distribute.ReduceOp.SUM, box_loss, axis=None))

        # update AP metric
        batch_pred_packs = pack_prediction(objectness, center, size, heading, label, class_probs,
                                           nms_type=config["nms_type"],
                                           inclass_nms=config["inclass_nms"],
                                           iou_threshold=config["nms_iou_threshold"],
                                           objectness_threshold=config["objectness_threshold"],
                                           keep_all_classes=config["keep_all_classes"])
        batch_gt_packs = pack_groundtruth(center=data_dict["box_center"],
                                          size=data_dict["box_size"],
                                          heading=data_dict["box_heading"],
                                          label=data_dict["box_label"],
                                          mask=data_dict["box_mask"])
        ap_metric.update(batch_pred_packs, batch_gt_packs)

    # compute metrics
    losses = total_loss_metric.result().numpy()
    vote_loss = vote_loss_metric.result().numpy()
    objectness_loss = obj_loss_metric.result().numpy()
    box_loss = box_loss_metric.result().numpy()
    reset_loss_metrics()
    P,R,AP = ap_metric.compute(iou_threshold=config["ap_iou_threshold"],
                               num_workers=config["num_worker"],
                               use_07_metric=config["use_07_metric"])
    ap_metric.reset()  #! reset after each evaluation epoch

    # loss log (summary step is mini-batch num, should align with train steps)
    sitter.scalar_summary("losses/vote_loss", vote_loss, writer="eval", reuse_last_step=True)
    sitter.scalar_summary("losses/objectness_loss", objectness_loss, writer="eval", reuse_last_step=True)
    sitter.scalar_summary("losses/box_loss", box_loss, writer="eval", reuse_last_step=True)

    sitter.logger.info(
        "eval loss - total: {:.5f}, vote: {:.5f}, objectness: {:.5f}, box: {:.5f}".format(
        losses, vote_loss, objectness_loss, box_loss)
    )

    # basic evaluation log (summary step is epoch num)
    mean_P = np.mean(list(P.values()))
    mean_R = np.mean(list(R.values()))
    mean_AP = np.mean(list(AP.values()))
    sitter.scalar_summary("basic/mean_precision", mean_P, writer="eval")
    sitter.scalar_summary("basic/mean_recall", mean_R, writer="eval")
    sitter.scalar_summary("basic/mean_AP", mean_AP,  writer="eval")
    sitter.info_log("eval basic - mean_P: {:.4f}, mean_R: {:.4f}, mean_AP: {:.4f}".format(mean_P, mean_R, mean_AP))

    # detailed evaluation log (summary step is epoch num)
    str_eval_summary_list = ["{}  \tprecision\t recall  \tAP @{:.2f}".format("classname".rjust(14), config["ap_iou_threshold"])]
    str_eval_summary_list.append("-" * 68)
    for label_id in sorted(P.keys()):
        class_name = id_to_name[label_id]
        p, r, ap = P[label_id], R[label_id], AP[label_id]
        sitter.scalar_summary(f"precision/{class_name}", p, writer="eval")
        sitter.scalar_summary(f"recall/{class_name}", r, writer="eval")
        sitter.scalar_summary(f"AP/{class_name}", ap, writer="eval")
        str_eval_summary_list.append("{}: \t{:.6f} \t{:.6f} \t{:.6f}".format(class_name.rjust(14), p, r, ap))
    sitter.info_log("eval detail - :\n" + "\n".join(str_eval_summary_list))

    return mean_AP  # as critical value for model-sitter


def train_main(total_epochs, epoch_offset=0):
    # log config info
    if epoch_offset == 0:
        sitter.info_log(">===== TRAINING CONFIG INFO =====<")
        config_strs = ["\nconfig: {"]
        for key in config:
            str_value = repr(config[key]).replace("\n", "\n" + " "*(8+len(key)))
            config_strs.append(f"    \"{key}\": {str_value},")
        config_strs.append("}")
        sitter.info_log("\n".join(config_strs))

    # collect meta
    bn_modules = [m for m in model.submodules if isinstance(m, keras.layers.BatchNormalization)]
    if epoch_offset == 0:  # training
        learning_rate = config["initial_lr"]
        bn_momentum = config["initial_bnm"]
        for bn_module in bn_modules: bn_module.momentum = 1.0 - bn_momentum  # init BN momentum
    else:  # refining
        learning_rate = keras.backend.get_value(optimizer.lr)
        bn_momentum = config["initial_bnm"] * config["bnm_decay_rate"] ** (epoch_offset // config["bnm_decay_period"])
    
    # main training loop
    for epoch in range(epoch_offset, total_epochs + epoch_offset):

        # schedule learning rate
        for thres, rate in zip(config["lr_decay_thres"], config["lr_decay_rates"]):
            if (epoch + 1) == thres:
                learning_rate = max(learning_rate * rate, config["minimal_lr"])
                keras.backend.set_value(optimizer.lr, learning_rate)
        sitter.scalar_summary("config/learning_rate", learning_rate, writer="train")

        # schedule batchnorm momentum
        if len(bn_modules) > 0:
            if epoch != 0 and epoch % config["bnm_decay_period"] == 0:
                bn_momentum = max(bn_momentum * config["bnm_decay_rate"], config["minimal_bnm"])
                for bn_module in bn_modules: bn_module.momentum = 1.0 - bn_momentum
            sitter.scalar_summary("config/bn_momentum", 1.0 - bn_momentum, writer="train")

        # train one epoch
        sitter.info_log(">===== EPOCH {0:^5d} =====<".format(epoch + 1))
        sitter.info_log("configs - lr:{:.6f}, bnm:{:.6f}".format(learning_rate, bn_momentum))
        train_one_epoch()

        # evaluate
        if (epoch + 1) % config["eval_period"] == 0:
            mean_AP = evaluate_one_epoch()
        else:
            mean_AP = None

        # notify sitter one epoch is end
        sitter.step(critical=mean_AP)


if __name__ == "__main__":
    # initialzie model weights
    with strategy.scope():  # for scannet use xyz, for sunrgbd use xyz-height
        model.build((None, config["downsample"], 3))

    # for refine only
    # with strategy.scope():
    #     sitter.restore_latest_ckpt()

    # alchemy
    train_main(config["max_epochs"], epoch_offset=sitter.overall_step)
