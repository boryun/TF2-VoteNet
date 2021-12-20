import os
import numpy as np
import cv2
import scipy.io as sciio
from utils import common_utils, data_utils, visual_utils


class META:
    # semantic color map, nyu40class label to 255 based RGB, following legend.png    
    color_map = {
        "bed": [158,0,142], "table": [122,71,130], "sofa": [119,77,0], "chair": [152,255,82], 
        "toilet": [133,169,0], "desk": [1,255,254], "dresser": [194,140,159], 
        "night_stand": [189,211,147], "bookshelf": [0,143,156], "bathtub": [187,136,0]
    }

    # 0-indexed id <-> class name
    class_to_nyuid = {
        "bed": 4, "table": 7, "sofa": 6, "chair": 5, "toilet": 33, "desk": 14, 
        "dresser": 17, "night_stand": 32, "bookshelf": 10, "bathtub": 36
    }
    nyuid_to_class = {v: k for k,v in class_to_nyuid.items()}

    # interested class for detection task
    detect_target_class = [
        "bed", "table", "sofa", "chair", "toilet", "desk", "dresser", "night_stand", 
        "bookshelf", "bathtub"
    ]
    detect_target_nyuid = np.array([4,7,6,5,33,14,17,32,10,36], dtype=np.int32)

    # 0-indexed id <-> target nyuid (label)
    id_to_label = {k:v for k,v in enumerate(detect_target_nyuid)}
    label_to_id = {v:k for k,v in id_to_label.items()}

    # max number of instance per scences contains
    MAX_BOXES = 45


class BoxInstance:
    def __init__(self, args_line):
        args = args_line.split(" ")
        for i in range(1, len(args)):
            args[i] = float(args[i])

        #! two diffenert between the following code and votenet source:
        #!   1. we times the box3d size by 2 so now the value is the actual length width and height instead of offset.
        #!   2. we didn't time the box3d_heading_angle by -1, so the angle remain anti-clockwise.
        self.classname = args[0]
        self.box2d = np.array([args[1], args[2], args[1]+args[3], args[2]+args[4]], dtype=np.float32)
        self.box3d_center = np.array([args[5], args[6], args[7]], dtype=np.float32)
        self.box3d_size = np.array([args[9], args[8], args[10]], dtype=np.float32) * 2.0  #! args[8] is width and args[9] is length!
        self.box3d_orientation = np.array([args[11], args[12], 0], dtype=np.float32)
        self.box3d_heading_angle = np.arctan2(self.box3d_orientation[1], self.box3d_orientation[0])


class Calibration:
    ''' Calibration matrices and utils
    
        We define five coordinate system in SUN RGBD dataset:
            1. depth coordinate: X-right, Y-forward, Z-up.
            2. camera coordinate: X-right, Y-down, Z-forward (camera X,Y,Z = depth X,-Z,Y).
            3. upright depth coordinate:  depth coordinate with Z aligned to gravity direction.
            4. upright camera coordinate: camera coordinate with -Y aligned to gravity direction.
            5. image coordinate: X(u)-right, Y(v)-down.

        Depth points and 3D bboxes label are stored in upright depth coordinate, 
        2D bboxes are stored in image coordinate.
    '''
    def __init__(self, args_lines):
        Rtilt = np.array([float(arg) for arg in args_lines[0].split(" ")], dtype=np.float32)
        self.Rtilt = np.reshape(Rtilt, (3, 3), order="F")  # the order="F" in this case is the same as Transpose
        
        K = np.array([float(arg) for arg in args_lines[1].split(" ")], dtype=np.float32)
        self.K = np.reshape(K, (3, 3), order="F")  # order="F" -> same as above

        self.f_u = self.K[0, 0]
        self.f_v = self.K[1, 1]
        self.c_u = self.K[0, 2]
        self.c_v = self.K[1, 2]
    
    def project_upright_depth_to_camera(self, pc):
        projected_pc = np.dot(pc[:,:3], self.Rtilt)
        projected_pc = self._flip_axis_to_camera(projected_pc)
        return projected_pc

    def project_upright_depth_to_image(self, pc):
        pc_camera = self.project_upright_depth_to_camera(pc)
        UV = np.dot(pc_camera, np.transpose(self.K))  # [N,3]
        UV[:,0] /= UV[:,2]
        UV[:,1] /= UV[:,2]
        return UV[:,:2], pc_camera[:,2]
    
    def project_image_to_camera(self, uv_depth):
        N = uv_depth.shape[0]
        x = ((uv_depth[:,0] - self.c_u) * uv_depth[:,2]) / self.f_u
        y = ((uv_depth[:,1] - self.c_v) * uv_depth[:,2]) / self.f_v
        pc_camera = np.zeros((N, 3))
        pc_camera[:,0] = x
        pc_camera[:,1] = y
        pc_camera[:,2] = uv_depth[:,2]
        return pc_camera

    def project_image_to_upright_camera(self, uv_depth):
        pc_camera = self.project_image_to_camera(uv_depth)
        pc_depth = self._flip_axis_to_camera(pc_camera)
        pc_upright_depth = np.dot(pc_depth, np.transpose(self.Rtilt))
        pc_upright_camera = self._flip_axis_to_camera(pc_upright_depth)
        return pc_upright_camera

    def project_upright_depth_to_upright_camera(self, pc):
        return self._flip_axis_to_camera(pc)

    def project_upright_camera_to_upright_depth(self, pc):
        return self._flip_axis_to_depth(pc)

    @staticmethod
    def _flip_axis_to_camera(pc):
        """
        Flip [X-right, Y-forward, Z-up] to [X-right, Y-down, Z-forward],
        (depth coordinate to camera coordinate).
        
        Args:
            pc: [N,3], original point cloud.
        Return:
            flipped_pc: [N,3], flipped point cloud.
        """
        # X'=X, Y'=-Z, Z'=Y
        flipped_pc = np.copy(pc)
        flipped_pc[:,[0,1,2]] = pc[:, [0,2,1]]
        flipped_pc[:,1] *= -1
        return flipped_pc

    @staticmethod
    def _flip_axis_to_depth(pc):
        """
        Flip [X-right, Y-down, Z-forward] to [X-right, Y-forward, Z-up],
        (camera coordinate to depth coordinate).

        Args:
            pc: [N,3], original point cloud.
        Return:
            flipped_pc: [N,3], flipped point cloud.
        """
        # X'=X, Y'=Z, Z'=-Y
        flipped_pc = np.copy(pc)
        flipped_pc[:,[0,1,2]] = pc[:, [0,2,1]]
        flipped_pc[:,2] *= -1
        return flipped_pc


class ExtractedReader:
    """
    Helper class for reading extracted data via matlab script.
    """
    def __init__(self, root_dir, use_v1=True):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "image")
        self.calib_dir = os.path.join(root_dir, "calib")
        self.depth_dir = os.path.join(root_dir, "depth")
        self.label_dir = os.path.join(root_dir, "label_v1" if use_v1 else "label")
    
    def get_image(self, idx):
        file_path = os.path.join(self.image_dir, "{:06d}.jpg".format(idx))
        return cv2.imread(file_path)

    def get_pc_depth(self, idx):
        file_path = os.path.join(self.depth_dir, "{:06d}.mat".format(idx))
        return sciio.loadmat(file_path)["instance"]
    
    def get_calibration(self, idx):
        file_path = os.path.join(self.calib_dir, "{:06d}.txt".format(idx))
        args_lines = common_utils.read_txt(file_path)
        calibration = Calibration(args_lines)
        return calibration
    
    def get_label_objects(self, idx):
        file_path = os.path.join(self.label_dir, "{:06d}.txt".format(idx))
        args_lines = common_utils.read_txt(file_path)
        objects = [BoxInstance(args_line) for args_line in args_lines]
        return objects   


def get_box3d_statistic(extracted_dir, mode, save_path=None):
    reader = ExtractedReader(extracted_dir)

    if mode == "train":
        data_index_list = common_utils.read_txt(os.path.join(extracted_dir, "train_data_idx.txt"))
    elif mode == "test":
        data_index_list = common_utils.read_txt(os.path.join(extracted_dir, "val_data_idx.txt"))
    elif mode == "all":
        data_index_list = []
        data_index_list.extend(common_utils.read_txt(os.path.join(extracted_dir, "train_data_idx.txt")))
        data_index_list.extend(common_utils.read_txt(os.path.join(extracted_dir, "val_data_idx.txt")))
    else:
        raise ValueError("Invalid mode, expect one of [\"train\", \"test\", \"all\"], got {}".format(mode))
    data_index_list = [int(idx) for idx in data_index_list]

    boxtype = []
    boxsize = []
    boxheading = []
    
    for idx in common_utils.get_tqdm(data_index_list):
        calib = reader.get_calibration(idx)
        objects = reader.get_label_objects(idx)

        for i, obj in enumerate(objects):
            if obj.classname not in META.detect_target_class:
                continue
            boxtype.append(obj.classname)
            boxsize.append(obj.box3d_size)
            boxheading.append(obj.box3d_heading_angle)
    
    if save_path is not None:
        common_utils.save_pickle(save_path, boxtype, boxsize, boxheading)
    
    # get average box size for different categories
    type_ids = np.array([META.class_to_nyuid[label] for label in boxtype], dtype=np.int32)
    size_ary = np.vstack(boxsize)
    for label in sorted(list(set(boxtype))):
        idx = np.where(type_ids == META.class_to_nyuid[label])
        mean_size = np.median(size_ary[idx], axis=0)
        print("{:s} = np.array([{:f}, {:f}, {:f}], dtype=np.float32)".format(label, *mean_size))


def preprocess(extracted_dir, output_dir, save_votes=False, use_v1=False, skip_emply_scene=False, maxpoints=None, unify=False):
    """
    Extract points, bounding box, and points vores for all scenes.

    Args:
        extracted_dir: path to the folder of data extracted via matlab script.
        output_dir: path to output folder.
        save_votes: whether to save votes for each points (keep up to 3).
        use_v1: whether to use v1 labels.
        skip_emply_scene: whether to skip scene with no box instance.
        maxpoints: maximum points to keep of each scene.
        unify: whether to ensure all scene has exact "maxpoints" points.
    Return:
        None
    """
    # extrace scene id
    data_index_list = []
    data_index_list.extend(common_utils.read_txt(os.path.join(extracted_dir, "train_data_idx.txt")))
    data_index_list.extend(common_utils.read_txt(os.path.join(extracted_dir, "val_data_idx.txt")))
    data_index_list = [int(idx) for idx in data_index_list]

    # create ouput folder
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # handle scene sequntially
    data_reader = ExtractedReader(root_dir=extracted_dir, use_v1=use_v1)
    data_index_iterator = common_utils.get_tqdm(data_index_list)
    for data_idx in data_index_iterator:
        # update tqdm postfix to current scene id (so we know which scene occurred an error, XD)
        data_index_iterator.set_postfix_str("current: {:06d}".format(data_idx))

        # save path
        pcd_path = os.path.join(output_dir, "{:06d}_pc.npz".format(data_idx))
        bbox_path = os.path.join(output_dir, "{:06d}_bbox.npy".format(data_idx))
        vote_path = os.path.join(output_dir, "{:06d}_vote.npz".format(data_idx))

        # skip processed
        if os.path.exists(pcd_path) and os.path.exists(bbox_path) and (not save_votes or os.path.exists(vote_path)):
            continue

        # extract and filter non valid instances (defined in META)
        objects = data_reader.get_label_objects(data_idx)
        objects = [obj for obj in objects if obj.classname in META.detect_target_class]

        if skip_emply_scene and len(objects) == 0:
            continue

        # collect&save point cloud (x-right, y-forward, z-uptop)
        pc = data_reader.get_pc_depth(data_idx)
        if maxpoints is not None and (unify or pc.shape[0] > maxpoints):
            pc = data_utils.random_sampling(pc, target_num=maxpoints)
        np.savez_compressed(pcd_path, pc=pc)

        # collect&save all oriented bounding boxes
        bboxes = np.zeros((len(objects), 8), dtype=np.float32)
        for i, obj in enumerate(objects):
            bboxes[i,0:3] = obj.box3d_center
            bboxes[i,3:6] = obj.box3d_size
            bboxes[i,6] = obj.box3d_heading_angle
            bboxes[i,7] = META.class_to_nyuid[obj.classname]
        np.save(bbox_path, bboxes)
    
        # collect votes for each points (i.e. deviation to the center of belonging instance)
        if save_votes:
            num_points = pc.shape[0]
            votes = np.zeros((num_points, 9), dtype=np.float32)  # 1 mask and 3 votes
            votes_mask = np.zeros((num_points,), dtype=np.float32)
            votes_count = np.zeros((num_points,), dtype=np.int32)  # idx of current vote, range(0,2)
            indices = np.arange(num_points)

            for i, obj in enumerate(objects):
                try:
                    box_corners = data_utils.get_box_corners(obj.box3d_center, obj.box3d_size, obj.box3d_heading_angle)
                    inbox_mask = data_utils.get_inbox_mask(pc, box_corners)

                    inbox_points = pc[inbox_mask, :]
                    votes_mask[inbox_mask] = 1

                    # calculate vote value for inbox points
                    deviation = np.expand_dims(obj.box3d_center, 0) - inbox_points[:,0:3]
                    sparse_inds = indices[inbox_mask]  # only keep inbox index

                    # update vote for each inbox points
                    for i, idx in enumerate(sparse_inds):
                        vote_idx = int(votes_count[idx])
                        votes[idx, vote_idx*3: (vote_idx+1)*3] = deviation[i,:]
                        if vote_idx == 0:  # copy the first vote to all three slot
                            votes[idx, 3:6] = deviation[i,:]
                            votes[idx, 6:9] = deviation[i,:]
                    votes_count[inbox_mask] = np.minimum(2, votes_count[inbox_mask] + 1)
                except Exception as e:
                    # Some scenes contains ill-conditioned bounding box, such that the box is 
                    # actually a 2-dimentional square or 1-dimentional point, which will cause 
                    # error when computing inbox points.
                    print("Error! scene:{:06d}, obj: {}, type:{}".format(data_idx, i + 1, e.__class__.__name__))
            np.savez_compressed(vote_path, votes=votes, mask=votes_mask)


def show_extracted_scene(scene_id, reader):
    """
    Visualize the point cloud, bounding box and heading of a extracted scene.

    Args:
        scene_id: integer index of scene.
        reader: ExtractedReader object in sunrgbd_utils.
    Return:
        None.
    """
    # point cloud
    ply = reader.get_pc_depth(scene_id)
    pcd = visual_utils.create_pointcloud(ply[:,:3], ply[:,3:6])

    # bounding boxes & heading line
    objects = reader.get_label_objects(scene_id)
    bboxes = []
    orientations = []
    for obj in objects:
        if obj.classname not in META.detect_target_class:
            continue
        
        # bounding box
        center = obj.box3d_center
        size = obj.box3d_size
        heading = obj.box3d_heading_angle
        color = META.color_map[obj.classname]
        bboxes.append(visual_utils.create_boundingbox(center, size, color, heading))

        # heading line
        heading_begin = center
        heading_end = center + obj.box3d_orientation
        orientations.append(visual_utils.create_lineset([heading_begin, heading_end], colors=color))

    print("number of objects: {}".format(len(bboxes)))

    axis_pcd = visual_utils.create_coordinate(size=1, origin=[0,0,0])
    visual_utils.draw_geometries(axis_pcd, pcd, *bboxes, *orientations)


def show_sunrgbd_gt(scene_id, data_dir):
    """
    Visualize a preporcessed scannet scene.

    Args:
        scene_id: integer idx of scene, e.g. 9960.
        data_dir: path to preprocessed data folder.
    Return:
        None.
    """
    file_prefix = os.path.join(data_dir, "{:06d}".format(scene_id))
    vertex = np.load(file_prefix + "_pc.npz")["pc"]
    bboxes = np.load(file_prefix + "_bbox.npy")
    vote_z = np.load(file_prefix + "_vote.npz")
    pts_votes = vote_z["votes"].astype(np.float32)
    pts_vote_masks = vote_z["mask"].astype(np.float32)
    print("scene: {}, num points: {}, num instances: {}".format(scene_id, vertex.shape[0], bboxes.shape[0]))

    pcd = visual_utils.create_pointcloud(points=vertex[:, :3], colors=vertex[:,3:6])

    obbs = []
    heading_vectors = []
    for i in range(bboxes.shape[0]):
        box = bboxes[i]
        center = box[0:3]
        size = box[3:6]
        heading = box[6]
        color = META.color_map[META.nyuid_to_class[int(box[7])]]
        obbs.append(visual_utils.create_boundingbox(center, size, color, heading))

        R = np.transpose(data_utils.rotZ(heading))
        head_from = center
        head_end = center + np.dot(np.array([3,0,0], np.float32), R)
        heading_vectors.append(visual_utils.create_lineset([head_from, head_end], colors=color))
    
    coordinate = visual_utils.create_coordinate(1.5)

    votes = []
    sample_num = 500
    valid_vote_idx = np.where(pts_vote_masks == 1.0)
    if len(valid_vote_idx[0]) > 0:
        sub_pts, sub_pts_votes = data_utils.random_sampling(vertex[valid_vote_idx], pts_votes[valid_vote_idx], target_num=sample_num)
        for i in range(sample_num):
            votes.append(visual_utils.create_lineset([sub_pts[i][:3], sub_pts[i][:3] + sub_pts_votes[i][0:3]], colors=[1,0,0]))
            votes.append(visual_utils.create_lineset([sub_pts[i][:3], sub_pts[i][:3] + sub_pts_votes[i][3:6]], colors=[0,1,0]))
            votes.append(visual_utils.create_lineset([sub_pts[i][:3], sub_pts[i][:3] + sub_pts_votes[i][6:9]], colors=[1,1,0]))

    visual_utils.draw_geometries(pcd, *obbs, *heading_vectors, coordinate, *votes)


if __name__ == "__main__":
    EXTRACT_SUNRGBD_DIR = "/media/boyu/depot/DataSet/SUNRGBD/SUNRGBD_RAW"
    OUTPUT_DIR = "datasets/SUNRGBD"

    preprocess(extracted_dir=EXTRACT_SUNRGBD_DIR,
               output_dir=OUTPUT_DIR,
               save_votes=True,
               use_v1=True,
               skip_emply_scene=False,
               maxpoints=50000,
               unify=False)

    #! known illed scene: 5097
    # show_sunrgbd_gt(5110, OUTPUT_DIR)

