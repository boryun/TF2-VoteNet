import os
import numpy as np
import plyfile
from scipy.spatial import ConvexHull, Delaunay
from utils.custom_op import grid_subsampling, nearest_neighbors  #! custom_op, inplace build required


# **************** #
#> PLY File Utils <#
# **************** #

def save_ply(filename, point, color=None, normal=None, label=None, custom=None, text=False):
    """
    Save specific point cloud as a .ply file. (Be awared: float color will
    be cast to int32, DO NOT pass normalized color value to this function.)
    
    Args:
        filename: path to the storage file with suffix of .ply.
        points: [N, 3] numpy array, xyz coordinate of N points.
        colors: [N, 3] numpy array, extra RGB info (range from 0 to 255).
        normals: [N, 3] numpy array, extra 3D normals.
        labels: [N,] numpy array, extra labels for each points.
        customs: list of (data, property) pairs which will be append to 
            'vertex' element of ply file, data is an N dimensional vector, 
            and property is a (name, dtype) pair.
        text: bool, .ply file format, True for ASCII and False for binary.
    Return:
        plydata: final ply data.

    For info about .ply file format: http://paulbourke.net/dataformats/ply
    """
    arrays = []
    dtypes = []

    arrays.extend(point.transpose())
    dtypes.extend([("x", "f4"), ("y", "f4"), ("z", "f4")])
    if color is not None:
        arrays.extend(color.transpose())
        dtypes.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    if normal is not None:
        arrays.extend(normal.transpose())
        dtypes.extend([("nx", "f4"), ("ny", "f4"), ("nz", "f4")])
    if label is not None:
        arrays.append(np.reshape(label, (-1,)))
        dtypes.append(("label", "u2"))
    if custom is not None:
        for data, dtype in custom:
            arrays.append(np.reshape(data, (-1,)))
            dtypes.append(dtype)
    vertex_element = np.rec.fromarrays(arrays, dtype=dtypes)
    plydata = plyfile.PlyData([plyfile.PlyElement.describe(vertex_element, "vertex")], text=text)

    dirname = os.path.dirname(filename)
    if os.path.isdir(dirname) and not os.path.exists(dirname):
        os.makedirs(dirname)
    plydata.write(filename)

    return plydata


def read_ply(file, point=True, color=False, normal=False, face=False, label=False, merge_label=False):
    """
    Load the required properties of given ply file to numpy array.

    Args:
        file: path to .ply file or a <plyfile.PlyData> object.
        point: bool, whether to load xyz coordinates. 
        color: bool, whether to load rgb value.
        normal: bool, whether to load normal vector.
        label: bool, whether to load label.
        merge_label: whether to merge label into the last row of data in 
            return value. (be awared: if merge label to data, label will 
            have dtype of float32)
    Return:
        data: [N, d] numpy array, d required properties for each N points,
            properties are in the sequence of x-y-z-r-g-b-nx-ny-nz.
        face: [M, 3] numpy array, 3 vertex index of each of M faces.
        label: [N,] numpy array, returned as needed.
    """
    if isinstance(file, str):  # relatively slow, as it will load the whole ply file.
        file = plyfile.PlyData.read(file)
    elif not isinstance(file, plyfile.PlyData):
        raise ValueError("expect file with type of str or plyfile.PlyData object, got {}".format(type(file)))

    ply_data, ply_face, ply_label = [], None, None
    
    if (point or color or normal or label) and "vertex" not in file:
        raise ValueError("given ply file doesn't contain vertex element!")
    if face and "face" not in file:
        raise ValueError("given ply file doesn't contain face element!")
    
    # extract each properties
    if point:
        ply_data.append(np.reshape(np.array(file["vertex"]["x"].data, dtype=np.float32), (-1, 1)))
        ply_data.append(np.reshape(np.array(file["vertex"]["y"].data, dtype=np.float32), (-1, 1)))
        ply_data.append(np.reshape(np.array(file["vertex"]["z"].data, dtype=np.float32), (-1, 1)))
    if color:
        ply_data.append(np.reshape(np.array(file["vertex"]["red"].data, dtype=np.float32), (-1, 1)))
        ply_data.append(np.reshape(np.array(file["vertex"]["green"].data, dtype=np.float32), (-1, 1)))
        ply_data.append(np.reshape(np.array(file["vertex"]["blue"].data, dtype=np.float32), (-1, 1)))
    if normal:
        ply_data.append(np.reshape(np.array(file["vertex"]["nx"].data, dtype=np.float32), (-1, 1)))
        ply_data.append(np.reshape(np.array(file["vertex"]["ny"].data, dtype=np.float32), (-1, 1)))
        ply_data.append(np.reshape(np.array(file["vertex"]["nz"].data, dtype=np.float32), (-1, 1)))
    if face:
        ply_face = np.vstack([f[0] for f in file["face"].data]).astype(np.int32)
    if label:
        ply_label = np.reshape(np.array(file["vertex"]["label"].data, dtype=np.int32), (-1,))
    
    if merge_label:
        ply_data.append(np.reshape(ply_label, (-1, 1)).astype(np.float32))

    if len(ply_data) > 1:
        ply_data = np.concatenate(ply_data, axis=1)

    # collect return values
    returns = []
    if point or color or normal or (label and merge_label):
        returns.append(ply_data)
    if face:
        returns.append(ply_face)
    if label and not merge_label:
        returns.append(ply_label)
    
    return returns if len(returns) > 1 else returns[0]


def inspect_ply(file):
    """
    Print the detail of given ply file.

    Args: 
        file: path to .ply file or a <plyfile.PlyData> object.
    Return:
        None
    """
    # raw ply file, acquire info from head lines
    if isinstance(file, str):
        # The number of properties is unknown until the end of the element block,
        # so we need a buffer to hold the slot until we got needed values.
        print_buffer = [""]
        ele_count = 0
        ele_name, num_items, num_props = "", "", 0
        has_unfilled_element = False

        with open(file, "rb") as f:
            line = str(f.readline(), encoding="utf-8").strip()
            while line != "" and "end_header" not in line:
                kwds = line.split(" ")

                if kwds[0] == "element":
                    ele_count += 1

                    # fill previous element string
                    if has_unfilled_element:
                        has_unfilled_element = False
                        print_buffer[-(num_props + 1)] = " "*4 + "{} ({}, {}):".format(ele_name, num_items, num_props)

                    # acquire current element info
                    has_unfilled_element = True
                    ele_name, num_items, num_props = kwds[1], kwds[2], 0
                    print_buffer.append("")
                
                elif kwds[0] == "property":
                    num_props += 1

                    if len(kwds) == 3:  # scalar
                        print_buffer.append(" "*8 + "\"{}\": scalar, {}".format(kwds[-1], kwds[1], ))
                    else:  # list
                        print_buffer.append(" "*8 + "\"{}\": list, {}".format(kwds[-1], kwds[-2], ))

                line = str(f.readline(), encoding="utf-8").strip()
            
            print_buffer[0] = "total elements: {}".format(ele_count)
            if has_unfilled_element:
                print_buffer[-(num_props + 1)] = " "*4 + "{} ({}, {}):".format(ele_name, num_items, num_props)
            
            for line in print_buffer:
                print(line)

    # plyfile.PlyData, acquire info from object
    else:
        if not isinstance(file, plyfile.PlyData):
            raise ValueError("expect file with type of str or plyfile.PlyData object, got {}".format(type(file)))

        print("total elements: {}".format(len(file)))
        for element in file.elements:
            # element name and ength
            print("    {} ({}, {}):".format(element.name, len(element), len(element.properties)))
            # properties info
            for property in element.properties:
                kwds = property.__str__().split()[1:]
                if(len(kwds) == 2):  # scalar property
                    print("        \"{}\": scalar, {}".format(kwds[-1], kwds[-2]))
                else:  # list property
                    print("        \"{}\": list, {}".format(kwds[-1], kwds[-2]))


def rasterize_mesh(vertices, faces, resolution):
    """ 
    Resterize a given mesh to point cloud. When handling dataset such as 
    scannet, points sampled from mesh will reasonably be a supplement to 
    the raw point cloud. (Yet "augment" the noise too XD)

    Args:
        vertices: [N, 3] numpy array, N points with xyz coordinates.
        faces: [M, 3] numpy array, M triangular faces with 3 vertex index 
            with respect to vertices param.
        resolution: minimal resolution for rasterize, each sampled point 
            corresponding to a resolution**2 square.
    Return:
        sampled_points: sampled points among all faces.
        closest_vertex: closest face vertex w.r.t. sampled points,used for 
            map properties from the original vertex to sampled points, e.g. 
            color, label...
    """
    faces3d = vertices[faces, :]  # [M,3,3], replace vertex index with vertex coordinates
    edges_vector = np.stack([faces3d[:, i, :] - faces3d[:, i - 1, :] for i in [2, 0, 1]], axis=1)  # [M,3,3] opposite edges of each vertex
    edges_length = np.linalg.norm(edges_vector, axis=2)  # [M,3] edge length

    sampled_points = []  # sampled points among all faces
    closest_vertex = []  # closest faces vertex w.r.t.  points

    for vertexs_idx, vertexs_xyz, edge_vector, edge_length in zip(faces, faces3d, edges_vector, edges_length):
        # shape of zips: (3,), (3,3), (3,3), (3,) 

        face_points = []  # sampled points for current face

        # skipe "fake" faces
        if np.min(edge_length) < 1e-9:
            continue
        
        # area is smaller than giver resolution is rasterized to one points (or just ignore it)
        if np.max(edge_length) < resolution:
            # get center point
            point = np.mean(edge_vector, axis=0)
            # find closest vertex
            dist = np.sum(np.square(vertexs_xyz - point), axis=1)
            vertex = vertexs_idx[np.argmin(dist)]
            # update global array
            sampled_points.append(point)
            closest_vertex.append(vertex)
            continue

        # form a local coordinate system
        O_idx = np.argmax(edge_length)  # origin of local coordinate system
        X_idx = (O_idx + 1) % 3  # index of X-axis direction vector
        Y_idx = (O_idx + 2) % 3  # index of Y-axis direction vector
        x_vec = -edge_vector[X_idx] / edge_length[X_idx]  # unit vector of x-axis
        y_vec = edge_vector[Y_idx] / edge_length[Y_idx]  # unit vector of y-axis
        # sampling points inside current face w.r.t. the local coordinate system
        x_span = np.arange((edge_length[X_idx] % resolution) / 2, edge_length[X_idx], resolution)
        y_span = np.arange((edge_length[Y_idx] % resolution) / 2, edge_length[Y_idx], resolution)
        x, y = np.meshgrid(x_span, y_span)
        inside_points = vertexs_xyz[O_idx, :] + np.expand_dims(x.ravel(), 1) * x_vec + np.expand_dims(y.ravel(), 1) * y_vec
        inside_points = inside_points[x.ravel() / edge_length[X_idx] + y.ravel() / edge_length[Y_idx] <= 1, :]
        face_points.append(inside_points)

        # sampling points on the edge
        for i in range(3):
            x_vec = edge_vector[i] / edge_length[i]
            O_idx = (i + 1) % 3
            x = np.arange((edge_length[i] % resolution) / 2, edge_length[i], resolution)
            edge_points = vertexs_xyz[O_idx, :] + np.expand_dims(x.ravel(), 1) * x_vec
            face_points.append(edge_points)

        # add face vertex to sampled points
        face_points.append(vertexs_xyz)

        # stack all sampled points and calculate closest vertex
        face_points = np.vstack(face_points)  # [num_samples, 3]
        dist_matrix = np.sum(np.square(np.expand_dims(face_points, axis=1) - vertexs_xyz), axis=2)  # [num_samples, 3]
        face_points_closest_vertex = np.argmin(dist_matrix, axis=1)  # [num_samples,]

        # update global samples
        sampled_points.append(face_points)
        closest_vertex.append(vertexs_idx[face_points_closest_vertex])
    
    return np.vstack(sampled_points).astype(np.float32), np.hstack(closest_vertex).astype(np.int32)


# ******************* #
#> Point Cloud Utils <#
# ******************* #

def random_sampling(*features, target_num):
    """
    Using random sampling to sampling from given data.

    Args:
        features: variable arguments, list of feature with any length.
        target_num: target upsample num.
    
    Return:
        sampled_feature: sampled feature list, has the same order as input.
    """
    current_num = features[0].shape[0]

    # validation
    if len(features) > 1:
        for feature in features:
            if feature.shape[0] != current_num:
                raise ValueError("All features should have same shape on dim axis.")

    # get samples index
    if current_num > target_num:  # downsampling
        sample_idx = np.random.choice(current_num, target_num, replace=False)
    else:  # upsampling, need to ensure that no feature get lost
        replace = False if (target_num - current_num <= current_num) else True
        sample_idx = np.random.choice(current_num, target_num - current_num, replace=replace)
        sample_idx = np.concatenate([np.arange(current_num), sample_idx], axis=0)
        # np.random.shuffle(sample_idx)  # optional
    
    # gather sampled features
    sampled_features = []
    for feature in features:
        sampled_features.append(feature[sample_idx, ...])
    
    return sampled_features if len(features) > 1 else sampled_features[0]


def grid_downsampling(points, features=None, labels=None, sampleDl=0.1):
    """
    Process grid subsampling on given point cloud. The subsampled point and
    features is calcualted by taking the mean values of points lays in the 
    same voxel, the label appeared most times is taken as the corresponding
    label of a voxel.

    Args:
        points: [N, 3], N points with xyz coordinate.
        features: [N, d], features of N points.
        classes: [N,] or [N, l], label(s) of each points.
        sampleDl: Side length of voxel.
    Return:
        subsampled_points: the subsampled points.
        subsampled_features: if features is provided.
        subsampled_labels: if labels is provided.
    """
    return grid_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl)


def knn_search(points, queries, K):
    """
    Get K-NN points for each query points within global point cloud.

    Args:
        points: [B, N, 3] numpy array, global point cloud.
        queries: [B, M, 3] numpy array, M queries for each batch.
        K: num neighbours.
    Return:
        nn_idx: [B, M, K] numpy array, K-NN index of each queries.
    """
    nn_idx = nearest_neighbors.knn_batch(points, queries, K, omp=True)
    return nn_idx


# ******************** #
#> Bounding Box Utils <#
# ******************** #

def rotX(angle):
    """
    Get rotate matrix along X-axis.

    Args:
        angle: rotate angle (anti-clockwise).
    Return:
        rot_mx: [3,3], rotate matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    rot_mx = np.array([[1,  0,  0],
                       [0,  c, -s],
                       [0,  s,  c]], dtype=np.float32)
    return rot_mx


def rotY(angle):
    """
    Get rotate matrix along Y-axis.

    Args:
        angle: rotate angle (anti-clockwise).
    Return:
        rot_mx: [3,3], rotate matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    rot_mx = np.array([[c,  0,  s],
                       [0,  1,  0],
                       [-s, 0,  c]], dtype=np.float32)
    return rot_mx


def rotZ(angle):
    """
    Get rotate matrix along Z-axis.

    Args:
        angle: rotate angle (anti-clockwise).
    Return:
        rot_mx: [3,3], rotate matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    rot_mx = np.array([[c, -s,  0],
                       [s,  c,  0],
                       [0,  0,  1]], dtype=np.float32)
    return rot_mx


def get_box_corners(center, size, heading_angle):
    """
    Generate 8 corners of single 3d bounding box.

    Args:
        center: [3,], center of bbox.
        size: [3], length, width and height of bbox.
        heading_angle: heaading angle of bbox.
    Return:
        corners: [8, 3], 8 corners of given box.
        i.e.
              1 -------- 0
             /|         /|
            2 -------- 3 . height (dz)
            | |        | |
            . 5 -------- 4
            |/         |/ width (dy)
            6 -------- 7
              length (dx)
        (x-right, y-forward, z-up)
    """
    # rotate matrix (left multiply)
    R = rotZ(heading_angle)

    # compute corner offsets
    l, w, h = size[0] / 2, size[1] / 2, size[2] / 2
    x_corners = np.array([l,-l,-l,l,l,-l,-l,l], dtype=np.float32)
    y_corners = np.array([w,w,-w,-w,w,w,-w,-w], dtype=np.float32)
    z_corners = np.array([h,h,h,h,-h,-h,-h,-h], dtype=np.float32)

    corners = np.vstack([x_corners, y_corners, z_corners])
    corners = np.dot(R, corners)
    corners[0,:] += center[0]
    corners[1,:] += center[1]
    corners[2,:] += center[2]
    
    return np.transpose(corners)


def batch_get_box_corners(center, size, heading_angle):
    """
    Generate 8 corners of 3d bounding box for batch data.

    Args:
        center: [B,N,3]
        size: [B,N,3]
        heading_angle: [B,N]

    Returns:
        corners: [B,N,8,3]
        i.e.
              1 -------- 0
             /|         /|
            2 -------- 3 . height (dz)
            | |        | |
            . 5 -------- 4
            |/         |/ width (dy)
            6 -------- 7
              length (dx)
        (x-right, y-forward, z-up)
    """

    # center->[B,N,3], size->[B,N,3], heading_angle->[B,N]
    batch_size = center.shape[0]
    num_proposal = center.shape[1]
    
    # generate batch rotate matrixs (right multiply)
    sin = np.sin(heading_angle)  # [B,N]
    cos = np.cos(heading_angle)  # [B,N]
    rot = np.zeros([batch_size, num_proposal, 3, 3], dtype=np.float32)  # [B,N,3,3]
    rot[:,:,0,0] = cos
    rot[:,:,1,0] = -sin
    rot[:,:,0,1] = sin
    rot[:,:,1,1] = cos
    rot[:,:,2,2] = 1

    # offsets for length, widht, height, [B,N,1]
    l = size[:, :, 0, np.newaxis] / 2
    w = size[:, :, 1, np.newaxis] / 2
    h = size[:, :, 2, np.newaxis] / 2
    
    # relative offsets for 8 corners
    corners = np.zeros([batch_size, num_proposal, 8, 3], dtype=np.float32)  # [B,N,8,3]
    corners[:,:,:,0] += np.concatenate([l,-l,-l,l,l,-l,-l,l], axis=2)
    corners[:,:,:,1] += np.concatenate([w,w,-w,-w,w,w,-w,-w], axis=2)
    corners[:,:,:,2] += np.concatenate([h,h,h,h,-h,-h,-h,-h], axis=2)
    corners = np.matmul(corners, rot)

    # add offsets to center point
    center = np.expand_dims(center, -1)
    corners[:,:,:,0] += center[:,:,0,:]
    corners[:,:,:,1] += center[:,:,1,:]
    corners[:,:,:,2] += center[:,:,2,:]

    return corners


def get_inbox_mask(pc, box_corners):
    """
    Generate a mask for whether a point is inside given bounding box
    or not of given point cloud.

    Args:
        pc: [N,3] point cloud.
        box_corners: [8,3], eight corners of bounding box.
    Return:
        mask: [N,], inbox mask of points.
    """
    hull = Delaunay(box_corners)
    mask = (hull.find_simplex(pc[:,:3]) >= 0)
    return mask


def nms_2d(boxes, scores, labels=None, threshold=0.25, old_type=False):
    """
    Non-maximum Suppression on 2D bounding box.

    Args:
        boxes: [N,4], bounding box parameters, 4 for the coordinate of the
            lower left corner and top right corner of bounding box.
        scores: [N,], scores for ranking, typically objectness.
        labels: if provided, only suppress the box with the same label when
            overlap occured.
        threshold: threshold value for filtering.
        old type: type of filtering, set false to use the fraction ofoverlap 
            area against the comparing box area and true for IoU value.
    Return:
        picked: list of picked boxes index.
    """
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)

    picked = []
    remains = np.argsort(scores)  # ascend order

    while(remains.size > 0):
        max_idx = remains[-1]
        picked.append(max_idx)
        remains = np.delete(remains, remains.size - 1)

        x_min = np.maximum(x1[max_idx], x1[remains])
        y_min = np.maximum(y1[max_idx], y1[remains])
        x_max = np.minimum(x2[max_idx], x2[remains])
        y_max = np.minimum(y2[max_idx], y2[remains])
        x_lap = np.maximum(0, x_max - x_min)
        y_lap = np.maximum(0, y_max - y_min)
        if old_type:
            overlap = (x_lap * y_lap) / areas[remains]
        else:
            overlap_areas = x_lap * y_lap
            overlap = overlap_areas / (areas[remains] + areas[max_idx] - overlap_areas)

        remains = np.delete(remains, np.where(overlap > threshold)[0])

    return picked


def nms_3d(boxes, scores, labels=None, threshold=0.25, old_type=False):
    """
    Non-maximum Suppression on 3D bounding box.

    Args:
        boxes: [N,6], bounding box parameters, 6 for the coordinate of the
            2 corners on the main diagonal.
        scores: [N,], scores for ranking, typically objectness.
        labels: if provided, only suppress the box with the same label when
            overlap occured.
        threshold: threshold value for filtering.
        old type: type of filtering, set false to use the fraction ofoverlap 
            area against the comparing box area and true for IoU value.
    Return:
        picked: list of picked boxes index.
    """
    x1, y1, z1 = boxes[:,0], boxes[:,1], boxes[:,2]
    x2, y2, z2 = boxes[:,3], boxes[:,4], boxes[:,5]
    areas = (x2 - x1) * (y2 - y1) * (z2 - z1)

    picked = []
    remains = np.argsort(scores)  # ascend order

    while(remains.size > 0):
        max_idx = remains[-1]
        picked.append(max_idx)
        remains = np.delete(remains, remains.size - 1)

        x_min = np.maximum(x1[max_idx], x1[remains])
        y_min = np.maximum(y1[max_idx], y1[remains])
        z_min = np.maximum(z1[max_idx], z1[remains])
        x_max = np.minimum(x2[max_idx], x2[remains])
        y_max = np.minimum(y2[max_idx], y2[remains])
        z_max = np.minimum(z2[max_idx], z2[remains])
        
        x_lap = np.maximum(0, x_max - x_min)
        y_lap = np.maximum(0, y_max - y_min)
        z_lap = np.maximum(0, z_max - z_min)

        if old_type:
            overlap = (x_lap * y_lap * z_lap) / areas[remains]
        else:
            overlap_areas = x_lap * y_lap * z_lap
            overlap = overlap_areas / (areas[remains] + areas[max_idx] - overlap_areas)
        
        if labels is not None:
            overlap = overlap * (labels[max_idx] == labels[remains])
        
        remains = np.delete(remains, np.where(overlap > threshold)[0])
    
    return picked


def polygon_area(polygon):
    """
    Calculate area of given polygon via "Shoelace formula", polygon is 
    defined as a set of points in a sequential order, e.g. clockwise or
    anti-clockwise.

    Args:
        polygon: [N,2] numpy array, sequential ordered points of polygon.
    Return:
        area: area of given polygon.
    """
    if len(polygon) == 0:
        return 0
    # extract x-index and y-index
    x = polygon[:,0]
    y = polygon[:,1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


def polygon_clip(subject_polygon, clip_polygon):
    """
    Calculate the intersect polygon of subject_polygon and clip_polygon. Polygon 
    is defined as a set of points in a sequential order, e.g. clockwise or anti-
    clockwise.

    Args:
        subject_polygon: [N,2] numpy array, sequential ordered points of polygon.
        clip_polygon: [N,2] numpy array, sequential ordered points of polygon.
    Return:
        intersection: [N,2] numpy array, sequential ordered points of polygon.
    """
    
    # calculate inner product of (p, cp2) and (cp2, cp1)
    def _inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
    # calculate intersection point of (cp1, cp2) and (s,e)
    def _computeIntersection():
        dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
        dp = [ s[0] - e[0], s[1] - e[1] ]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0] 
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    # loop over all edge of clipping polygon, (cp1-cp2 is vertex of edge)
    outputList = subject_polygon
    cp1 = clip_polygon[-1]
    for clipVertex in clip_polygon:
        cp2 = clipVertex
        inputList = outputList

        # loop over all edge of subject polygon, (s-e is vertex of edge)
        outputList = []
        s = inputList[-1]
        for subjectVertex in inputList:
            e = subjectVertex
            if _inside(e):
                if not _inside(s):
                    outputList.append(_computeIntersection())
                outputList.append(e)
            elif _inside(s):
                outputList.append(_computeIntersection())
            s = e
        cp1 = cp2

        # terminate if to polygons not intersect
        if len(outputList) == 0:
            break

    intersection = np.array(outputList, dtype=np.float32)
    return intersection


def compute_iou(box1_corners, box2_corners):
    """
    Calculate the IoU of given boxes, assume the corners of each box has
    the following order:
          1 -------- 0
         /|         /|
        2 -------- 3 . height (dz)
        | |        | |
        . 5 -------- 4
        |/         |/ width (dy)
        6 -------- 7
          length (dx)
    (x-right, y-forward, z-up)
    
    Args:
        box1_corners: [8,3] numpy array, 8 corners of box1.
        box2_corners: [8,3] numpy array, 8 corners of box2.
    Return:
        iou_2d: iou of bottom polygon.
        iou_3d: iou of 3d box.
    """
    # extract bottom polygon
    polygon1 = box1_corners[4:,:2]
    polygon2 = box2_corners[4:,:2]

    # IoU 2D
    inter_polygon = polygon_clip(polygon1.tolist(), polygon2.tolist())
    # inter_area = ConvexHull(inter_polygon).volume if len(inter_polygon) > 0 else 0
    inter_area = polygon_area(inter_polygon)
    area1 = polygon_area(polygon1)
    area2 = polygon_area(polygon2)
    iou_2d = inter_area / (area1 + area2 - inter_area)

    # IoU 3D
    z_max = min(box1_corners[0,2], box2_corners[0,2])
    z_min = max(box1_corners[4,2], box2_corners[4,2])
    inter_volume = inter_area * max(0.0, z_max - z_min)
    volume1 = (box1_corners[0,2] - box1_corners[4,2]) * area1
    volume2 = (box2_corners[0,2] - box2_corners[4,2]) * area2
    iou_3d = inter_volume / (volume1 + volume2 - inter_volume)

    return iou_3d, iou_2d


if __name__ == "__main__":
    # ply = plyfile.PlyData.read("datasets/demo/scene.ply")
    # vertex = ply['vertex'].data
    # faces = ply['face'].data
    # xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    # face = np.vstack([ary[0] for ary in faces])
    # sp, cv = rasterize_mesh(xyz, face, 0.01)
    # rgb = np.vstack([vertex['red'][cv], vertex['green'][cv], vertex['blue'][cv]]).T
    # alpha = vertex['alpha'][cv]
    # save_ply("datasets/demo/scene_sampled.ply", sp, rgb, customs=[(alpha, ("alpha", "u1"))], text=True)
    pass

