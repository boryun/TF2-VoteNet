import numpy as np
# open3d 0.13.0 required "/lib/x86_64-linux-gnu/libm.so.6" which may not 
# exist in some linux releases, e.g. ubuntu 16.04 LTS, however if we only
# need to train the model, it is ok to ignore such error. 
try:
    import open3d as o3d
except Exception as e:
    print(f"\033[0;93mError occured when importing open3d: {e}, ignore this if you don't need visualization.\033[0m")

# -------------------------------------------------------- #
# Reference: http://www.open3d.org/docs/release/index.html #
# -------------------------------------------------------- #

def create_pointcloud(points, colors):
    """
    Create an open3d PointCloud object.

    Args:
        points: [N,3] numpy array, XYZ coordinate of points.
        colors: [N,3] numpy array, RGB colors of points.
    Return:
        pcd: open3d PointCloud object.
    """
    colors = np.asarray(colors, dtype=np.float32)
    if np.max(colors) > 1:
        colors /= 255.0
    if len(colors.shape) == 1:
        colors = [colors for _ in range(len(points))]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def create_boundingbox(center, size, color, orientation=None):
    """
    Create an open3d OrientedBoundingBox object.

    Args:
        center: [3,] array, XYZ coordinate of box center.
        size: [3,] array, width, length and height of box.
        color: [3,] array, normalized RGB color.
        orientation: 1 scalar, box orientation (rotation angle of Z axis).
    Return:
        bbox: open3d OrientedBoundingBox object.
    """
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = size
    bbox.color = color if np.max(color) <= 1 else np.array(color) / 255.
    if orientation is not None:
        bbox.R = bbox.get_rotation_matrix_from_xyz([0, 0, orientation])
    return bbox


def create_lineset(points, lines=None, colors=None):
    """
    Create an open3d LineSet object of given points and lines, if lines
    is None, the points will connect one by one sequentially.

    Args:
        points: list of 3d end points for lines.
        lines: list of idx pair of points for each line.
        colors: RGB colors for each line.
    Return:
        linset: open3d LineSet object.
    """
    if lines is None:
        lines = [[i, i+1] for i in range(len(points)-1)]
    if colors is None:
        colors = [0, 0, 0]

    colors = np.asarray(colors, dtype=np.float32)
    if np.max(colors) > 1:
        colors /= 255.0
    if len(colors.shape) == 1:
        colors = [colors for _ in range(len(lines))]
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)

    return lineset


def create_coordinate(size=1, origin=[0,0,0]):
    """
    Create an open3d TriangleMesh object which constitute a coordinate 
    frame.

    Args:
        size: object size.
        origin: origin of frame.
    Return:
        mesh_cord: TriangleMesh object of coord frame.
    """
    mesh_cord = o3d.geometry.TriangleMesh.create_coordinate_frame(size, origin)
    return mesh_cord


def draw_geometries(*objects, **kwargs):
    """
    Visualize open3d objects.

    Args:
        objects: variable list of open3d geometry objects.

    Commonly used keyword args:
        window_name: The displayed title of the visualization window.
        width: The width of the visualization window.
        height: The height of the visualization window.
        left: The left margin of the visualization window.
        top: The top margin of the visualization window.        

    Return:
        None.
    """
    o3d.visualization.draw_geometries(objects, **kwargs)


if __name__ =="__main__":
    pass
