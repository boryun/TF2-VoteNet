import os
import json
import numpy as np
import pandas as pd
from utils import common_utils, data_utils, visual_utils


class META:
    # semantic color map, nyu40class label to 255 based RGB, following legend.png
    color_map = {
        "unclassified": [0,0,0], "wall": [190,153,112], "floor": [189,198,255],"cabinet": [213,255,0], 
        "bed": [158,0,142], "chair": [152,255,82],"sofa": [119,77,0], "table": [122,71,130], "door": [0,174,126],
        "window": [0,125,181], "bookshelf": [0,143,156], "picture": [107,104,130], "counter": [255,229,2], 
        "desk": [1,255,254], "curtain": [255,166,254], "refridgerator": [232,94,190], "shower curtain": [0,100,1], 
        "toilet": [133,169,0], "sink": [149,0,58], "bathtub": [187,136,0], "otherfurniture": [0,0,255]
    }

    # nyu40id <-> nyu40class, all ids: 0-40, 50, 149
    nyuid_to_class = {
        0: "unclassified", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa", 
        7: "table", 8: "door", 9: "window", 10: "bookshelf", 11: "picture", 12: "counter", 
        14: "desk", 16: "curtain", 24: "refridgerator", 28: "shower curtain", 33: "toilet", 
        34: "sink", 36: "bathtub", 39: "otherfurniture"
    }
    class_to_nyuid = {v:k for k, v in nyuid_to_class.items()}

    # interested class for detection task
    detect_target_nyuid = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39], dtype=np.int32)

    # 0-indexed id <-> target nyuid (label)
    id_to_label = {k:v for k,v in enumerate(detect_target_nyuid)}
    label_to_id = {v:k for k,v in id_to_label.items()}

    # max number of instance per scences contains
    MAX_BOXES = 54


def read_labelmap(tsv_file, label_from="raw_category", label_to="nyu40id"):
    """
    Get mapping of 'label_from' to 'label_to' from tsv file.

    Args:
        tsv_file: tsv_file path.
        label_from: str name of key.
        label_to: str name of value.
    Return:
        label_map: dict of <label_from. label_to>.
    """
    csv = pd.read_csv(tsv_file, delimiter="\t", header=0)
    label_map = {}
    for i in range(len(csv)):
        label_map[csv.iloc[i][label_from]] = csv.iloc[i][label_to]
    return label_map


def read_meta(meta_file, key):
    """
    Get meta value from scene meta file ('<scene_id>.txt').

    Args:
        meta_file: path to meta file of any scene.
        keys: string or list of required meta properties.
    Return:
        returns: values w.r.t. given keys.
    """
    # read meta lines
    with open(meta_file, "r", encoding="UTF-8") as file:
        meta_lines = file.readlines()
    
    # create meta dict
    meta_dict = {}
    for line in meta_lines:
        if "=" in line:
            args = line.split("=")
            meta_dict[args[0].strip()] = args[1].strip()
    
    if isinstance(key, str):
        return meta_dict[key]
    else:
        return [meta_dict[k] for k in key]


def read_aggregation(agg_file):
    """
    Read aggregation file and return <id/label, segment_idx> dicts.

    Args:
        agg_file: path to <scene_id>.aggregation.json file.
    Return:
        obj_id_to_segs: dict of <object_id, segments>.
        label_to_segs: dict of <label, segments>.
    """
    obj_id_to_segs = {}
    label_to_segs = {}

    with open(agg_file, "r") as file:
        data = json.load(file)

    seg_groups = data["segGroups"]
    for i in range(len(seg_groups)):
        obj_id = seg_groups[i]["objectId"] + 1
        label = seg_groups[i]["label"]
        segs = seg_groups[i]["segments"]
        obj_id_to_segs[obj_id] = segs
        if label in label_to_segs:
            label_to_segs[label].extend(segs)
        else:
            label_to_segs[label] = segs

    return obj_id_to_segs, label_to_segs


def read_segmentation(seg_file):
    """
    Read segmentation file and return <segment_id, vertices> dict.

    Args:
        seg_file: path to <scene_id>._vh_clean_2.0.010000.segs.json file.
    Return:
        seg_to_verts: <segment_id, vertices> dict.
        num_vertex: number of vertex (with semantic labels) in current secene.
    """
    seg_to_verts = {}

    with open(seg_file, "r") as file:
        data = json.load(file)
    
    seg_indices = data['segIndices']
    num_vertex = len(seg_indices)

    for i in range(num_vertex):
        seg_id = seg_indices[i]
        if seg_id in seg_to_verts:
            seg_to_verts[seg_id].append(i)
        else:
            seg_to_verts[seg_id] = [i]

    return seg_to_verts, num_vertex


def read_scene_data(scene_dir, label_map):
    """
    Read scene data of specific scene.

    Args:
        scene_dir: path to scene folder, e.g. ScanNet/scans/scene0000_00.
        label_map: dict map semantic name to unique id (e.g. nyu40id).
    Return:
        points: [N,6],  points of current scene, 6 for x-y-z-r-g-b.
        label_ids: [N,], semantic class id, 0 for unannotated.
        instance_ids: [N,], corresponding bbox id, 0 for non-instance points.
        instance_bboxes: [num_instances, 7], bbox parameters, 7 for x, y, z,
            length(dx), width(dy), height(dz), and semantic_id sequentially. 
    """
    # input file path
    scene_name = os.path.basename(scene_dir)
    input_prefix = os.path.join(scene_dir, scene_name)
    ply_file = input_prefix + "_vh_clean_2.ply"
    agg_file = input_prefix + ".aggregation.json"
    seg_file = input_prefix + "_vh_clean_2.0.010000.segs.json"
    meta_file = input_prefix + ".txt"  # includes axisAlignment info for the train set scans. 

    # read scene data
    points = data_utils.read_ply(ply_file, point=True, color=True)  # [N, 6], xyzrgb
    obj_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)

    # align scene points to axis
    alignment = read_meta(meta_file, "axisAlignment")
    align_matrix = np.fromstring(alignment, dtype=np.float32, sep=" ").reshape(4, 4)
    temp_xyz = np.ones((points.shape[0], 4), dtype=np.float32)
    temp_xyz[:,:3] = points[:,:3]
    temp_xyz = np.dot(temp_xyz, align_matrix.transpose())
    points[:,:3] = temp_xyz[:,:3]

    # collect label_id for each points (nyu40id semantic id)
    label_ids = np.zeros(shape=(num_verts,), dtype=np.uint32)  # 0 for unannotated
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    
    # collect instance_id for each points (i.e. bounding box id, limited within each scene)
    instance_ids = np.zeros(shape=(num_verts,), dtype=np.uint32)  # 0 for unannotated, or non-instance point.
    obj_id_to_label_id = {}
    for obj_id, segs in obj_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = obj_id
            if obj_id not in obj_id_to_label_id:
                obj_id_to_label_id[obj_id] = label_ids[verts[0]]

    # collect axis aligned bounding boxes
    instance_bboxes = np.zeros(shape=(len(obj_id_to_label_id.keys()), 9), dtype=np.float32)
    for obj_id in obj_id_to_segs:
        label_id = obj_id_to_label_id[obj_id]

        obj_points = points[instance_ids==obj_id, :3]
        if len(obj_points) == 0:
            continue
        xmin = np.min(obj_points[:, 0])
        ymin = np.min(obj_points[:, 1])
        zmin = np.min(obj_points[:, 2])
        xmax = np.max(obj_points[:, 0])
        ymax = np.max(obj_points[:, 1])
        zmax = np.max(obj_points[:, 2])
        bbox = np.array([
            (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2,  # center points
            xmax - xmin, ymax - ymin, zmax - zmin,  # length, width, height (dx, dy, dz)
            0, label_id, obj_id  # heading angle, semantic_id, instance_id
        ], dtype=np.float32)

        instance_bboxes[obj_id-1, :] = bbox

    return points, label_ids, instance_ids, instance_bboxes


def preprocess(scan_dir, output_dir, save_votes=False, exclude_sem=None, valid_ins=None, maxpoints=None, unify=False):
    """
    Extract scene vertex, semantic id of each points, instance id (bbox id) for each
    points, and scene bounding box parameters of each scene in ScanNet dataset.

    Args:
        scan_dir: path so folder contains scene data, e.g. ScanNet/scans.
        output_dir: path to storage folder.
        save_votes: whether to save votes for each points.
        exclude_sem: semantic id of points need to be exclude from scene.
        valid_ins: semantic id of bbox need to be keep.
        maxpoints: maximum points to keep of each scene.
        unify: whether to ensure all scene has exact "maxpoints" points.
    Return:
        None.
    """
    # load label_map and all scene name
    dir_path = os.path.dirname(os.path.abspath(__file__))
    label_map = read_labelmap(os.path.join(dir_path, "meta/scannetv2-labels.combined.tsv"), 
                              label_from="raw_category", label_to="nyu40id")
    target_scenes = []
    target_scenes.extend(common_utils.read_txt(os.path.join(dir_path, "meta/scannetv2_train.txt")))
    target_scenes.extend(common_utils.read_txt(os.path.join(dir_path, "meta/scannetv2_val.txt")))

    # make output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # process each scene
    for scene_name in common_utils.get_tqdm(target_scenes):
        output_prefix = os.path.join(output_dir, scene_name)
        pts_path = output_prefix + "_vertex.npy"
        sem_path = output_prefix + "_sem_id.npy"
        ins_path = output_prefix + "_ins_id.npy"
        bbox_path = output_prefix + "_bbox.npy"
        vote_path = output_prefix+ "_vote.npz"

        # skip processed
        if os.path.exists(pts_path) and os.path.exists(sem_path) and os.path.exists(bbox_path) and (not save_votes or os.path.exists(vote_path)):
            continue

        # extrace and filter scene data
        vertices, sem_ids, ins_ids, bboxes = read_scene_data(os.path.join(scan_dir, scene_name), label_map)
        
        if exclude_sem is not None:
            mask = np.logical_not(np.in1d(sem_ids, exclude_sem))
            vertices = vertices[mask, :]
            sem_ids = sem_ids[mask]
            ins_ids = ins_ids[mask]
        
        if valid_ins is not None:
            mask = np.in1d(bboxes[:, 7], valid_ins)
            bboxes = bboxes[mask, :]

        # subsample **after** filtering
        if maxpoints is not None and (unify or vertices.shape[0] > maxpoints):
            vertices, sem_ids, ins_ids = data_utils.random_sampling(vertices, sem_ids, ins_ids, target_num=maxpoints)
        
        # save sceene data  
        np.save(pts_path, vertices)
        np.save(sem_path, sem_ids)
        np.save(ins_path, ins_ids)
        np.save(bbox_path, bboxes)

        # generate votes
        if save_votes:
            votes = np.zeros((vertices.shape[0], 3), dtype=np.float32)
            votes_mask = np.zeros((vertices.shape[0],), dtype=np.float32)
            for i in range(bboxes.shape[0]):
                cur_ins_id = bboxes[i, 8]
                cur_ins_center = bboxes[i,0:3]
                ins_points_id = np.where(ins_ids == cur_ins_id)
                votes[ins_points_id,:] = cur_ins_center - vertices[ins_points_id, 0:3]
                votes_mask[ins_points_id] = 1.0
            np.savez(vote_path, votes=votes, mask=votes_mask)


def show_scannet_gt(scene_id, data_dir):
    """
    Visualize a preporcessed scannet scene.

    Args:
        scene_id: name of scene, e.g. scene0000_00.
        data_dir: path to preprocessed data folder.
    Return:
        None.
    """
    file_prefix = os.path.join(data_dir, scene_id)
    vertex = np.load(file_prefix + "_vertex.npy")
    bboxes = np.load(file_prefix + "_bbox.npy")
    vote_z = np.load(file_prefix + "_vote.npz")
    pts_votes = vote_z["votes"].astype(np.float32)  # [N,3]
    pts_vote_masks = vote_z["mask"].astype(np.float32)  # [N,]
    print("scene: {}, num points: {}, num instances: {}".format(scene_id, vertex.shape[0], bboxes.shape[0]))

    pcd = visual_utils.create_pointcloud(points=vertex[:, :3], colors=vertex[:,3:6])
    boxes = []
    for i in range(bboxes.shape[0]):
        box = bboxes[i]
        center = box[0:3]
        size = box[3:6]
        color = META.color_map[META.nyuid_to_class[int(box[7])]]
        boxes.append(visual_utils.create_boundingbox(center, size, color))
    
    votes = []
    sample_num = 1000
    valid_vote_idx = np.where(pts_vote_masks == 1.0)
    if len(valid_vote_idx[0]) > 0:
        sub_pts, sub_pts_votes = data_utils.random_sampling(vertex[valid_vote_idx], pts_votes[valid_vote_idx], target_num=sample_num)
        for i in range(sample_num):
            votes.append(visual_utils.create_lineset([sub_pts[i][:3], sub_pts[i][:3] + sub_pts_votes[i][:3]], colors=[1,0,0]))

    coord = visual_utils.create_coordinate(2)

    visual_utils.draw_geometries(pcd, *boxes, coord, *votes)


if __name__ == "__main__":
    SCANNET_DIR = "/media/boyu/depot/DataSet/ScanNet/scans"  # raw dataset dir
    OUTPUT_DIR = "datasets/ScanNet"  # output dir

    preprocess(scan_dir=SCANNET_DIR, 
               output_dir=OUTPUT_DIR,
               save_votes=True,
               valid_ins=META.detect_target_nyuid,
               maxpoints=50000,
               unify=False)

    # show_scannet_gt("scene0000_00", OUTPUT_DIR)
