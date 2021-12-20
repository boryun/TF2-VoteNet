import os
import numpy as np
import tensorflow as tf
from utils import common_utils, data_utils
from utils.dataset_utils.ScanNet.scannet_utils import META


def rot_scene_type_1(vertex, bboxes, votes, angle):
    """
    Rotate the point cloud and bounding box along Z-axis of given angle
    while keep the box size fixed and adjust box heading.
    """
    rot_mat = np.transpose(data_utils.rotZ(angle))

    vertex[:,0:3] = np.dot(vertex[:,0:3], rot_mat)
    bboxes[:,0:3] = np.dot(bboxes[:,0:3], rot_mat)
    bboxes[:,6] += angle
    votes = np.dot(votes, rot_mat)

    return vertex, bboxes, votes


def rot_scene_type_2(vertex, bboxes, votes, angle):
    """
    Rotate the point cloud and bounding box along Z-axis of given angle
    while keep the box heading fix (as 0) and adjust box size.
    """
    # rotate vertex
    rot_mat = np.transpose(data_utils.rotZ(angle))
    vertex[:,0:3] = np.dot(vertex[:,0:3], rot_mat)

    # rotate box center
    bboxes[:,0:3] = np.dot(bboxes[:,0:3], rot_mat)

    # adjust box size
    x_offset = bboxes[:,3] / 2
    y_offset = bboxes[:,4] / 2

    corners_x = np.zeros((bboxes.shape[0], 4), dtype=np.float32)
    corners_y = np.zeros((bboxes.shape[0], 4), dtype=np.float32)
    corners_3d = np.zeros((bboxes.shape[0], 3), dtype=np.float32)

    for i, corner in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):
        corners_3d[:,0] = corner[0] * x_offset
        corners_3d[:,1] = corner[1] * y_offset
        corners_3d = np.dot(corners_3d, rot_mat)
        corners_x[:,i] = corners_3d[:,0]
        corners_y[:,i] = corners_3d[:,1]

    bboxes[:,3] = 2.0*np.max(corners_x, axis=1)
    bboxes[:,4] = 2.0*np.max(corners_y, axis=1)

    # rotate votes
    votes = np.dot(votes, rot_mat)

    return vertex, bboxes, votes


def augment_pipline(vertex, box_params, votes, inplace=True):
    if not inplace:
        vertex = vertex.copy()
        box_params = box_params.copy()
        votes = votes.copy()

    # flip along the YZ plane
    if np.random.rand() > 0.5:
        vertex[:,0] = -1 * vertex[:,0]
        box_params[:,0] = -1 * box_params[:,0]
        votes[:,0] = votes[:,0] * -1

    # flip along the XZ plane
    if np.random.rand() > 0.5:
        vertex[:,1] = -1 * vertex[:,1]
        box_params[:,1] = -1 * box_params[:,1]
        votes[:,1] = votes[:,1] * -1

    # random rotate (-5,5) degree along Z-axis
    rot_angle = (np.random.random()*np.pi/18) - np.pi/36
    vertex, box_params, votes = rot_scene_type_2(vertex, box_params, votes, rot_angle)

    # rescale point cloud
    # scale_ratio = np.random.rand() * 0.3 + 0.85
    # scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
    # vertex[:,0:3] *= scale_ratio
    # box_params[:,0:3] *= scale_ratio
    # box_params[:,3:6] *= scale_ratio
    # votes[:,:3] *= scale_ratio

    return vertex, box_params, votes


def read_preprocessed_data(data_dir, scene_id):
    # load data from file
    file_prefix = os.path.join(data_dir, scene_id)
    vertex = np.load(file_prefix + "_vertex.npy").astype(np.float32)  # [N, 6], xyz(3), rgb(3)
    bboxes = np.load(file_prefix + "_bbox.npy").astype(np.float32)  # [N, 8], center(3), size(3), heading(1), nyuid(1)
    vote_z = np.load(file_prefix + "_vote.npz")
    votes = vote_z["votes"].astype(np.float32)  # [N,3]
    vote_mask = vote_z["mask"].astype(np.float32)  # [N,]

    # normalize RGB channels
    vertex[:,3:6] /= 255.0

    # convert nyu40id to 0-indexed id
    for i in range(bboxes.shape[0]):
        bboxes[i, 7] = META.label_to_id[bboxes[i, 7]]

    return vertex, bboxes, votes, vote_mask


def get_dataset_generator(scene_list, data_dir, box_padding, shuffle=False, augment=False, downsample=None, cache_data=False):
    # data cache (ignored if cache_data is False)
    cache_dict = {}

    # input data generator
    def generator():
        if shuffle:
            np.random.shuffle(scene_list)

        for scene in scene_list:
            # prepare pre-processed data
            if not cache_data:
                points, bboxes, votes, vote_mask = read_preprocessed_data(data_dir, scene)
            elif scene not in cache_dict:
                points, bboxes, votes, vote_mask = read_preprocessed_data(data_dir, scene)
                cache_dict[scene] = (points, bboxes, votes, vote_mask)
            else:
                points, bboxes, votes, vote_mask = cache_dict[scene]
            
            # if cache data, make a copy of original data in cace of inplace modification
            if cache_data:
                points, bboxes, votes, vote_mask = points.copy(), bboxes.copy(), votes.copy(), vote_mask.copy()

            # downsample || shuffle
            if downsample is not None:
                ds_idx = np.random.choice(points.shape[0], downsample, replace=downsample>points.shape[0])
                points = points[ds_idx, ...]
                votes = votes[ds_idx, ...]
                vote_mask = vote_mask[ds_idx, ...]
            else:
                idx = np.arange(len(points))
                np.random.shuffle(idx)
                points = points[idx, ...]
                votes = votes[idx, ...]
                vote_mask = vote_mask[idx, ...]

            # data augmentation
            if augment:
                points, bboxes, votes = augment_pipline(points, bboxes, votes)

            # ensure heading lying in [-np.pi, np.pi]
            # bboxes[np.where(bboxes[:,6] < -np.pi), 6] += 2*np.pi
            # bboxes[np.where(bboxes[:,6] >  np.pi), 6] -= 2*np.pi

            # only keep xyz of points
            points = points[:,:3]  # [N,3]

            # unify the size of bboxes ary
            unified_bboxes = np.zeros((box_padding, bboxes.shape[1]), dtype=np.float32)
            unified_bboxes_mask = np.zeros((box_padding,), dtype=np.float32)
            unified_bboxes[:bboxes.shape[0], :] = bboxes
            unified_bboxes_mask[:bboxes.shape[0]] = 1.0

            # input dict
            data_value = {
                "points": points,
                "point_votes": np.tile(votes, (1,3)),
                "point_votes_mask": vote_mask,
                "box_center": unified_bboxes[:,0:3],
                "box_size": unified_bboxes[:,3:6],
                "box_heading": unified_bboxes[:,6],
                "box_label": unified_bboxes[:,7],
                "box_mask": unified_bboxes_mask
            }

            yield data_value

    # input dtype
    data_dtype = {
        "points": tf.float32,
        "point_votes": tf.float32,
        "point_votes_mask": tf.float32,
        "box_center": tf.float32,
        "box_size": tf.float32,
        "box_heading": tf.float32,
        "box_label": tf.int32,
        "box_mask": tf.float32
    }

    # input shape
    data_shape = {
        "points": (None, 3),
        "point_votes": (None, 9),
        "point_votes_mask": (None,),
        "box_center": (None, 3),
        "box_size": (None, 3),
        "box_heading": (None,),
        "box_label": (None,),
        "box_mask": (None,)
    }

    return generator, data_dtype, data_shape


def get_dataset(data_dir, max_instances=64, shuffle=False, augment=True, downsample=None, split="train", cache_data=False):
    """
    Get tensorflow dataset object for train and test.
    """
    # load split scene
    dir_path = os.path.dirname(os.path.abspath(__file__))
    if split == "train":
        data_index_list = common_utils.read_txt(os.path.join(dir_path, "meta/scannetv2_train.txt"))
    elif split == "val":
        data_index_list = common_utils.read_txt(os.path.join(dir_path, "meta/scannetv2_val.txt"))
    else:
        raise ValueError("expect split to be one of [\"train\", \"val\"], got {}".format(split))

    # get dataset generator
    generator, dtypes, dshapes = get_dataset_generator(scene_list=data_index_list,
                                                       data_dir=data_dir,
                                                       box_padding=max_instances,
                                                       shuffle=shuffle,
                                                       augment=augment,
                                                       downsample=downsample,
                                                       cache_data=cache_data)

    # create tf dataset object
    dataset = tf.data.Dataset.from_generator(generator, dtypes, dshapes)
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(data_index_list)))

    return dataset


if __name__ == "__main__":
    OUTPUT_DIR = "datasets/ScanNet"

    ds = get_dataset(OUTPUT_DIR, max_instances=60, shuffle=False, augment=False, downsample=50000, split="train")
    ds = ds.batch(8, drop_remainder=False).cache()
    dsit = iter(ds)
    res = next(dsit)

    from utils import visual_utils
    def show_data(idx):
        pts = res["points"][idx].numpy()
        box_center = res["box_center"][idx].numpy()
        box_size = res["box_size"][idx].numpy()
        box_heading = res["box_heading"][idx].numpy()
        box_label = res["box_label"][idx].numpy()
        box_mask = res["box_mask"][idx].numpy()
        pts_votes = res["point_votes"][idx].numpy()
        pts_vote_masks = res["point_votes_mask"][idx].numpy()

        pcd = visual_utils.create_pointcloud(pts[:,0:3], pts[:,3:6])
        boxes = []
        headings = []
        for i in range(int(np.sum(box_mask))):
            center = box_center[i,:]
            size = box_size[i,:]
            heading = box_heading[i]
            label = box_label[i]
            color = META.color_map[META.nyuid_to_class[META.id_to_label[label]]]
            boxes.append(visual_utils.create_boundingbox(center, size, color, heading))
            R = np.transpose(data_utils.rotZ(heading))
            headings.append(visual_utils.create_lineset([center, center + np.dot(np.array([2,0,0]), R)], colors=color))

        votes = []
        sample_num = 1000
        valid_vote_idx = np.where(pts_vote_masks == 1.0)
        if len(valid_vote_idx[0]) > 0:
            sub_pts, sub_pts_votes = data_utils.random_sampling(pts[valid_vote_idx], pts_votes[valid_vote_idx], target_num=sample_num)
            for i in range(sample_num):
                votes.append(visual_utils.create_lineset([sub_pts[i][:3], sub_pts[i][:3] + sub_pts_votes[i][:3]], colors=[1,0,0]))

        coord = visual_utils.create_coordinate(2)

        visual_utils.draw_geometries(pcd, *boxes, *headings, coord, *votes)

    # show_data(0)
    # show_data(1)
    # show_data(2)
    # show_data(3)
    # show_data(4)
    # show_data(5)
    # show_data(6)
    # show_data(7)