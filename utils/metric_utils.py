import numpy as np
from multiprocessing import Pool


def confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Calculate confusion matrix of given predicts and labels.
    
    Args:
        y_true: [N,] numpy array, N ground truth labels.
        y_pred: [N,] numpy array, N network predict labels.
        num_classes: total number of classes.
    Return:
        cm: [num_classes, num_classes] numpy array, confusion matrix.
    """
    y_true = np.reshape(y_true, (-1,))
    y_pred = np.reshape(y_pred, (-1,))
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred))
    cm_flatten = np.bincount(num_classes * y_true + y_pred, minlength=num_classes**2)
    cm = np.reshape(cm_flatten, (num_classes, num_classes))
    return cm


def average_precision(recalls, precisions, use_07_metric=False):
    """
    Calculate the average precision of given precision-recall curve. i.e.
    an estimation of the area under the P-R curve. (note that, we usually
    smooth the P-R curve by replace each precision value with the maximum
    precision on the right of the corresponding recall level, i.e. each
    precision is taken as "precision[i] = np.max(precision[i:])".)

    - For PASCALL VOC2007, the AP is estimate by calculating the mean 
      precision of 11 uniformly distributed values on the recall axis, 
      i.e. (0, 0.1, ..., 0.9, 1.0), the precision corresponding to each 
      recall is estimated as the maximum precisions among those whose 
      recalls larger than current one.

    - For later methodology (VOC2010-2012), the AP is taken as the AUC
      (area under curve) of the "smoothed" P-R curve.
    
    Args:
        recalls: [N,], an array of N recall values.of P-R curve.
        precisions: [N,], an array of N precision values of P-R curve.
        use_07_metric: use PASCALL VOC2007 metric, default False.
    Return:
        ap: average precision computed on given P-R value pairs.
    """
    assert len(recalls) == len(precisions)

    # estimate AP over 11 spines
    if use_07_metric:
        ap = 0.0
        for r in np.arange(0.0, 1.1, 0.1):
            idx = np.where(recalls >= r)[0]
            if(len(idx) > 0):
                ap += np.max(precisions[idx]) / 11.0

    # estimate AP over all recalls
    else:
        # append left and right boundary
        recalls = np.concatenate([[0], recalls, [1]], axis=0)
        precisions = np.concatenate([[0], precisions, [0]], axis=0)

        # "smooth" the P-R curve
        for i in range(len(recalls)-1, 0, -1):
            precisions[i-1] = np.maximum(precisions[i-1], precisions[i])

        # calculate area for each indivadual bins
        idx = np.where(recalls[:-1] != recalls[1:])[0]  # take unique recalls (to skip "empty" bins)
        areas = (recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]

        ap = np.sum(areas)
    
    return ap


class MetricBox:
    """
    A sample metric class, support accuracy, perclass-accuracy, MeanIoU, and
    MacroF1 calculations, all calculations are base on confusion matrix.
    """
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((num_class, num_class), dtype=np.int64)

    def update(self, y_true, y_pred):
        y_true = np.reshape(y_true, (-1,))
        y_pred = np.reshape(y_pred, (-1,))
        cm_flatten = np.bincount(self.num_class * y_true + y_pred, minlength=self.num_class**2)
        self.confusion_matrix += np.reshape(cm_flatten, (self.num_class, self.num_class))
    
    def compute_acc(self):
        correct = np.sum(np.diag(self.confusion_matrix))
        total = np.sum(self.confusion_matrix)

        with np.errstate(divide="ignore", invalid="ignore"):
            acc = correct / total
        acc = np.nan_to_num(acc, copy=False, nan=0)

        return acc
    
    def compute_acc_perclass(self):
        correct = np.diag(self.confusion_matrix)
        total = np.sum(self.confusion_matrix, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            perclass_acc = correct / total
        perclass_acc = np.nan_to_num(perclass_acc, copy=False, nan=0)

        return perclass_acc

    def compute_iou(self):
        TP = np.diag(self.confusion_matrix)
        TP_FP = np.sum(self.confusion_matrix, axis=0)
        TP_FN = np.sum(self.confusion_matrix, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            IoU = TP / (TP_FP + TP_FN - TP)
        IoU = np.nan_to_num(IoU, copy=False, nan=0)
        
        return IoU

    def compute_miou(self):
        IoU = self.compute_iou()
        return np.mean(IoU)
    
    def compute_macrof1(self):
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1 = 2 * (precision * recall) / (precision + recall)
        np.nan_to_num(F1, copy=False, nan=0)
        return np.mean(F1)

    def get_confusion_matrix(self):
        return self.confusion_matrix.copy()
    
    def reset(self, num_class=None):
        if num_class is not None:
            self.num_class = num_class
            self.confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.double)
        else:
            self.confusion_matrix[:,:] = 0


class APMetric:
    def __init__(self, iou_func, pred_unpack_func=None, gt_unpack_func=None):
        self.scene_id = 0  # scene_count & uique id for current scene
        self.pred_dict = {}  # {"<label_id>": {"<scene_id>": [(ins_params, score), ...]}}
        self.gt_dict = {}  # {"<label_id>": {"<scene_id>": [ins_params, ...]}}

        if pred_unpack_func is None:
            pred_unpack_func = lambda x: x
        if gt_unpack_func is None:
            gt_unpack_func = lambda x: x
        
        self.iou_func = iou_func  # calculate IoU between two instance
        self.pred_unpack_func = pred_unpack_func  # unpack the pred_pack to (score, label, instance)
        self.gt_unpack_func = gt_unpack_func  # unpack the gt_pack to (label, instance)
    
    def update(self, batch_pred_packs, batch_gt_packs):
        # loop over result of each batch
        for pred_packs, gt_packs in zip(batch_pred_packs, batch_gt_packs):
            # append preds to pred_dict
            for pack in pred_packs:
                score, label, instance = self.pred_unpack_func(pack)

                # init pred_dict keys
                if label not in self.pred_dict:
                    self.pred_dict[label] = {}
                if self.scene_id not in self.pred_dict[label]:
                    self.pred_dict[label][self.scene_id] = []

                # init gt_dict keys (incase some class is "created" in prediction)
                if label not in self.gt_dict:
                    self.gt_dict[label] = {}
                if self.scene_id not in self.gt_dict[label]:
                    self.gt_dict[label][self.scene_id] = []
                
                # append instance to corresponding list
                self.pred_dict[label][self.scene_id].append((score, instance))
            
            # append gts to GT dict
            for pack in gt_packs:
                label, instance = self.gt_unpack_func(pack)

                # init gt_dict keys
                if label not in self.gt_dict:
                    self.gt_dict[label] = {}
                if self.scene_id not in self.gt_dict[label]:
                    self.gt_dict[label][self.scene_id] = []

                # init pred_dict keys (incase some class is missed in prediction)
                if label not in self.pred_dict:
                    self.pred_dict[label] = {}
                # if self.scene_id not in self.pred_dict[label]:
                #     self.pred_dict[label][self.scene_id] = []
                
                # append instance to corresponding list
                self.gt_dict[label][self.scene_id].append(instance)
            
            self.scene_id += 1
    
    def compute(self, iou_threshold=0.25, num_workers=1, use_07_metric=False):
        # result dict, map label to corresponding value
        precision_dict = {}
        recall_dict = {}
        ap_dict = {}

        if num_workers <= 1:
            for label in self.gt_dict.keys():
                precision, recall, ap = self.inclass_evaluate(pred=self.pred_dict[label], 
                                                            gt=self.gt_dict[label], 
                                                            iou_func=self.iou_func, 
                                                            iou_threshold=iou_threshold,
                                                            use_07_metric=use_07_metric)
                precision_dict[label] = precision
                recall_dict[label] = recall
                ap_dict[label] = ap
        else:
            # create iterables of args for each worker
            args_iter = []
            for label in self.gt_dict.keys():
                args_iter.append((self.pred_dict[label], self.gt_dict[label], self.iou_func, iou_threshold, use_07_metric))

            # process pool
            with Pool(processes=num_workers) as pool:
                returns = pool.starmap(self.inclass_evaluate, args_iter)

            # collect result 
            for idx, label in enumerate(self.gt_dict.keys()):
                precision_dict[label] = returns[idx][0]
                recall_dict[label] = returns[idx][1]
                ap_dict[label] = returns[idx][2]
        
        return precision_dict, recall_dict, ap_dict

    def reset(self):
        self.scene_id = 0
        self.pred_dict = {}
        self.gt_dict = {}

    @staticmethod
    def inclass_evaluate(pred, gt, iou_func, iou_threshold, use_07_metric=False):
        # detection_dict map each scene_id to a detection dict:
        # {"<scene_id>": {
        #       "ins": list of params (for IoU calculation) of each instance.
        #       "det": list of bool mask indicate whether an instance is detected or not.
        # }}
        detection_dict = {}
        total_instances = 0
        # collect GT instances
        for scene_id in gt.keys():
            ins = np.array(gt[scene_id], dtype=np.float32)
            det = [False] * len(ins)
            total_instances += len(ins)
            detection_dict[scene_id] = { "ins": ins, "det": det }
        # pad empty scene
        for scene_id in pred.keys():
            if scene_id not in gt:
                detection_dict[scene_id] = { "ins": np.array([]), "det": [] }

        # extract individual predicts
        scene_ids = []  # unique id indicate the scene of each predict instance
        confidences = []  # instance objectness
        det_instances = []  # bounding box instances
        for scene_id in pred.keys():
            for score, instance in pred[scene_id]:
                scene_ids.append(scene_id)
                confidences.append(score)
                det_instances.append(instance)
        confidences = np.array(confidences, dtype=np.float32)
        det_instances = np.array(det_instances, dtype=np.float32)

        # sort instances by cofidences
        sort_idx = np.argsort(-confidences)
        scene_ids = [scene_ids[i] for i in sort_idx]
        confidences = confidences[sort_idx]
        det_instances = det_instances[sort_idx, ...]
        
        # traverse all predictions to acquire TP and FP
        num_detections = len(scene_ids)
        TP = np.zeros(num_detections, dtype=np.float32)
        FP = np.zeros(num_detections, dtype=np.float32)
        for d in range(num_detections):
            D = detection_dict[scene_ids[d]]
            pred_box = det_instances[d, ...]
            gt_boxes = D["ins"]
            max_iou = -np.inf  # actually 0 will just work..
            match_idx = -1

            # compute IoU against each GT instance within predict scene
            for i in range(len(gt_boxes)):
                iou = iou_func(pred_box, gt_boxes[i,...])
                if iou > max_iou:
                    max_iou = iou
                    match_idx = i
                        
            #update result
            if max_iou > iou_threshold:
                if not D["det"][match_idx]:
                    TP[d] = 1.0
                    D["det"][match_idx] = True
                else:  #! regard redundant detection as False Positive
                    FP[d] = 1.0
            else:
                FP[d] = 1.0
        
        # compute precision, recall
        fp_cum = np.cumsum(FP)  # [N,]
        tp_cum = np.cumsum(TP)  # [N,]
        # ignore 0-divide errors and correct the NaN or Inf result caused by them later
        with np.errstate(divide="ignore", invalid="ignore"):
            recall = tp_cum / total_instances  # nan or inf will occur if total_instances is 0 (an unused label is predicted)
            precision = tp_cum / (tp_cum + fp_cum)  # nan or inf will occur if first few elements of tp_cum is 0
        np.nan_to_num(recall, copy=False, nan=0, posinf=0, neginf=0)
        np.nan_to_num(precision, copy=False, nan=0, posinf=0, neginf=0)

        P = precision[-1] if len(precision) > 0 else 0
        R = recall[-1] if len(recall) > 0 else 0
        AP = average_precision(recall, precision, use_07_metric=use_07_metric)

        return P, R, AP


if __name__ == "__main__":
    pass