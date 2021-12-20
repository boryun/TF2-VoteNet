# parse args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=str, default="0", help="Idx of GPU to use (split with ','). [default: 0]")
parser.add_argument("--dataset", type=str, default="scannet", help="Dataset name. scannet or sunrgbd. [default: scannet]")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size. [default: 16]")
parser.add_argument("--log_dir", type=str, default="logs", help="directory of saved logs and checkpoints. [default: logs]")
parser.add_argument("--keep_all_classes", action="store_true", help="Whether to keep all classes of class with maximum prob when parse proposals.")
args = parser.parse_args()


# config
config = {
    # basic config
    "dataset": args.dataset,
    "downsample": 40000,
    "detect_proposals": 256,

    # test config
    "gpu": [int(idx) for idx in args.gpus.split(",")],
    "batch_size": args.batch_size,
    "prefetch": 4,

    # metric config
    "nms_type": 1,  # 0: 2D, 1: 3D
    "inclass_nms": True,
    "nms_iou_threshold": 0.25,
    "objectness_threshold": 0.05,
    "ap_iou_thresholds": [0.25, 0.5],
    "keep_all_classes": args.keep_all_classes,
    "use_07_metric": False,
    "num_worker": 8,

    # log config
    "log_dir": args.log_dir,
}


# headers
import numpy as np
import tensorflow as tf
from model.network import DetectModel
from model.util import parse_prediction, pack_prediction, pack_groundtruth, unpack_prediction, unpack_groundtruth, box3d_iou
from utils.metric_utils import APMetric
from utils.trainval_utils import ModelSitter
from utils.common_utils import get_tqdm


# Ensure TensorFlow only use memory of specific GPUs
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

id_to_name = { k:META.nyuid_to_class[META.id_to_label[k]] for k in META.id_to_label.keys() }


# generate dataset
test_dataset = get_dataset(data_dir=config["dataset_dir"],
                            max_instances=config["max_instances"],
                            shuffle=False,
                            augment=False,
                            downsample=config["downsample"],
                            split="val",
                            cache_data=False)
test_dataset = test_dataset.batch(config["batch_size"], False).prefetch(config["prefetch"])
test_epoch_steps = len(test_dataset)


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
    train_dataset = strategy.experimental_distribute_dataset(test_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    sitter = ModelSitter(config["log_dir"], model=model, optimizer=None, mode="eval")

ap_metric = APMetric(box3d_iou, pred_unpack_func=unpack_prediction, gt_unpack_func=unpack_groundtruth)


# val functions
@tf.function
def evaluate_one_step(data_dict):
    pred_dict = model(inputs=data_dict["points"][:,:,:3], training=False)
    objectness, center, size, heading, label = parse_prediction(
        pred_dict,
        num_heading_bin=config["num_heading_bin"],
        mean_size=config["mean_size"],
        parse_heading=config["parse_heading"]
    )
    class_probs = tf.nn.softmax(pred_dict["scores"], axis=2)
    return objectness, center, size, heading, label, class_probs


def evaluate_main():
    # predict
    for data_dict in get_tqdm(test_dataset, total=test_epoch_steps):
        # run one evaluating step with strategy
        objectness, center, size, heading, label, class_probs = strategy.run(evaluate_one_step, args=(data_dict,))

        # aggregate values from all device (for latest TF, one can just use strategy.gather)
        for key in data_dict:
            data_dict[key] = tf.concat(strategy.experimental_local_results(data_dict[key]), axis=0)
        objectness = tf.concat(strategy.experimental_local_results(objectness), axis=0)
        center = tf.concat(strategy.experimental_local_results(center), axis=0)
        size = tf.concat(strategy.experimental_local_results(size), axis=0)
        heading = tf.concat(strategy.experimental_local_results(heading), axis=0)
        label = tf.concat(strategy.experimental_local_results(label), axis=0)
        class_probs = tf.concat(strategy.experimental_local_results(class_probs), axis=0)

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

    # compute metric
    for iou_threshold in config["ap_iou_thresholds"]:
        P,R,AP = ap_metric.compute(iou_threshold=iou_threshold,
                                num_workers=config["num_worker"],
                                use_07_metric=config["use_07_metric"])

        # title
        sitter.info_log(f">>> AP @ {iou_threshold} <<<".center(56))

        # basic evaluation log
        mean_P = np.mean(list(P.values()))
        mean_R = np.mean(list(R.values()))
        mean_AP = np.mean(list(AP.values()))
        sitter.info_log("basic - mean_P: {:.4f}, mean_R: {:.4f}, mean_AP: {:.4f}".format(mean_P, mean_R, mean_AP))

        # detailed evaluation log
        str_eval_summary_list = ["{}  \tprecision\t recall  \tAP @{:.2f}".format("classname".rjust(14), iou_threshold)]
        str_eval_summary_list.append("-" * 68)
        for label_id in sorted(P.keys()):
            class_name = id_to_name[label_id]
            p, r, ap = P[label_id], R[label_id], AP[label_id]
            str_eval_summary_list.append("{}: \t{:.6f} \t{:.6f} \t{:.6f}".format(class_name.rjust(14), p, r, ap))
        sitter.info_log("detail - :\n" + "\n".join(str_eval_summary_list) + "\n")


# the output is stored in "<path to log dir>/eval_log.txt", the log is append to existing log 
# file by default, remove the log file manually if you want to clear old eval info, and, also
#! BE AWARE as we are doing downsample on-the-fly, so "LUCK" also affect the eval result, XD
if __name__ == "__main__":
    # initialzie model weights
    with strategy.scope():
        model.build((None, config["downsample"], 3))
    sitter.restore_latest_ckpt()

    # dump evaluation configs
    sitter.info_log(">===== EVAL CONFIG INFO =====<")
    config_strs = ["\nconfit: {"]
    for key in config:
        str_value = repr(config[key]).replace("\n", "\n" + " "*(8+len(key)))
        config_strs.append(f"    \"{key}\": {str_value},")
    config_strs.append("}")
    sitter.info_log("\n".join(config_strs))
    
    # evaluate
    evaluate_main()

