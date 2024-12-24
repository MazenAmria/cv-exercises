import os
import tarfile
import io
import logging
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.spatial.transform import Rotation
from PIL import Image

import open3d as o3d

log = logging.getLogger(__name__)
np.set_printoptions(suppress=True)

target_path = Path(".")
if not (target_path / "marker").is_dir():
    url = "https://lmb.informatik.uni-freiburg.de/lectures/computer_vision_I/exercisedata/marker.tar"
    print(f"Downloading {url}")
    res = requests.get(url, timeout=30)
    data = res.content
    data_io = io.BytesIO(data)
    print(f"Extracting...")
    with tarfile.TarFile(fileobj=data_io, mode="r") as tar:
        tar.extractall(path=target_path.as_posix())
    print(f"Done")
else:
    print(f"Folder exists: {target_path.as_posix()}/marker")

class ViewLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        assert os.path.isdir(base_path)
        files = sorted(os.listdir(self.base_path))
        files = [f for f in files if (f.startswith("rgb_") and f.endswith(".png"))]
        self.max_idx = int(files[-1].replace("rgb_", "").replace(".png", ""))
        print(f"Loaded {self.max_idx+1} images.")

    def __len__(self):
        return self.max_idx + 1

    def __getitem__(self, idx):
        return self.get_rgbdp(idx)

    def get_info(self):
        # read given parameters from info.json
        info_path = os.path.join(self.base_path, "info.json")
        with open(info_path, "rb") as f_obj:
            info = json.load(f_obj)
        return info

    def get_intrinsics(self):
        # get intrintic camera matrix
        info = self.get_info()
        calib = info["camera"]["calibration"]
        return calib

    def get_K(self):
        calib = self.get_intrinsics()
        cam_intrinsic = np.eye(3)
        cam_intrinsic[0, 0] = calib["fx"]
        cam_intrinsic[1, 1] = calib["fy"]
        cam_intrinsic[0, 2] = calib["ppx"]
        cam_intrinsic[1, 2] = calib["ppy"]
        return cam_intrinsic

    def get_robot_pose(self, idx, return_dict=False):
        # read robot pose, convert to 4x4 homogeneous format
        # returns T_robot_tcp
        pose_file = os.path.join(self.base_path, "pose_{0:04d}.json".format(idx))
        with open(pose_file, "rb") as f_obj:
            pose = json.load(f_obj)

        # pose contains position xyz, rotation xyz and depth (7 parameters)
        pose_m = np.eye(4)
        pose_m[:3, :3] = Rotation.from_euler(
            "xyz", [pose[x] for x in ["rot_x", "rot_y", "rot_z"]]
        ).as_matrix()
        pose_m[:3, 3] = [pose[x] for x in ["x", "y", "z"]]
        if return_dict:
            return pose_m, pose
        else:
            return pose_m

    def get_rgb_file(self, idx):
        # read RGB image
        rgb_file = os.path.join(self.base_path, "rgb_{0:04d}.png".format(idx))
        return rgb_file

    def get_depth_file(self, idx):
        # read depth image
        depth_file = os.path.join(self.base_path, "depth_{0:04d}.png".format(idx))
        return depth_file

    def get_rgbdp(self, idx):
        # get RGB, scaled-depth, robot pose
        rgb_file = self.get_rgb_file(idx)
        rgb = np.asarray(Image.open(rgb_file))

        pose_m, pose_d = self.get_robot_pose(idx, True)

        # depth images here are saved 16-bit grayscale for efficient storage
        # depth_scaling is the factor required to convert depth to meters
        depth_file = self.get_depth_file(idx)
        depth_scaling = pose_d["depth_scaling"]
        depth = np.asarray(Image.open(depth_file), dtype=np.float32) * depth_scaling
        return rgb, depth, pose_m

    def get_cam_pose(self, idx, marker_dir="pose_marker_one"):
        # get camera pose in 4x4 homogeneous format
        # these are the results from marker detection: T_cam_marker
        marker_dir = os.path.join(self.base_path, marker_dir)
        fn = "{0:08d}.json".format(idx)
        pose_fn = os.path.join(marker_dir, fn)
        with open(pose_fn, "r") as fo:
            T_cam_marker = np.array(json.load(fo))
        return T_cam_marker

    def get_projection_matrix(self):
        # returns a 3x4 projection matrix using the intrinsics
        # it projects a 3d point in in homogeneous coordinates
        # to a 2d point in homogeneous coordinates
        # assuming the camera frame and world frame are aligned

        # START TODO #################
        K = self.get_K()
        cam_mat = np.hstack((K, np.zeros((3, 1))))
        # END TODO ###################
        assert cam_mat.shape == (3, 4)
        return cam_mat

    def project(self, X):
        if X.shape[0] == 3:
            # convert coordinate to homogeneous if it is euclidean
            if len(X.shape) == 1:
                X = np.append(X, 1)
            else:
                X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)

        # project a point from the 3d camera coordinate system into
        # the 2d camera frame. (in homogeneous coordinates)
        # given the camera projection matrix
        x = self.get_projection_matrix() @ X

        # convert homogeneous to euclidean and check if it is inside the image boundaries
        result = np.round(x[0:2] / x[2]).astype(int)
        width, height = self.get_intrinsics()["width"], self.get_intrinsics()["height"]
        if not (0 <= result[0] < width and 0 <= result[1] < height):
            log.warning("Projected point outside of image bounds")
        return result[0], result[1]


vl = ViewLoader(base_path="marker")
print("camera calibration:")
camera_calibration = vl.get_K()
K = np.array(camera_calibration)
print(K.round(2))

from matplotlib.widgets import Slider

fig, ax = plt.subplots(1)
image, depth, pose = vl.get_rgbdp(0)
line = ax.imshow(np.asarray(image))
ax.set_axis_off()


def update(w):
    image, depth, pose = vl.get_rgbdp(w)
    line.set_data(np.asarray(image))
    fig.canvas.draw_idle()


ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03])  # Position of the slider
slider = Slider(ax_slider, 'Index', 0, len(vl) - 1, valinit=0, valstep=1)

slider.on_changed(update)

plt.show()

from PIL import ImageDraw


def show_marker_pose(image, T_cam_marker, cam_proj):
    """
    draw the coordinate frame into each image for which we have detection results

    Arguments:
        image: image as numpy.ndarray
        T_cam_marker: shape (4, 4), transform from marker into camera
            x_cam = T_cam_marker @ x_marker
    Returns:
        im: image (should be PIL.Image.Image)
    """
    print(T_cam_marker)
    # START TODO #################
    # using PIL.ImageDraw
    # 1. Define 4 points in 3D homogeneous coordinates: <x, y, z, center>
    #    (The center of the coordinate system and a point for each of 3 axes.)
    #    Note that the 3D here are specified in meters.
    # 2. Transform the <x, y, z, center> coordinates into the camera frame
    # 3. Project the homogeneous coordinates <cam_x, cam_y, cam_z, center>
    #    to the camera image (euclidean)
    # 4. Draw one line for each axis
    x_marker = np.array([
        [0., 0., 0., 1.],
        [1., 0., 0., 1.],
        [0., 1., 0., 1.],
        [0., 0., 1., 1.],
    ])

    x_cam = T_cam_marker @ x_marker.T

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)

    x_proj = cam_proj @ x_cam

    x_image = []
    for i in range(x_proj.shape[1]):
        x, y, z = x_proj[:, i]
        u = x / z
        v = y / z
        x_image.append((u, v))

    origin = x_image[0]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, color in enumerate(colors):
        if origin:
            draw.line([origin, x_image[i + 1]], fill=color, width=10)
    # END TODO ###################

    # type(im) should be PIL.Image.Image
    return im


image, depth, robot_pose = vl.get_rgbdp(1)
T_cam_marker = vl.get_cam_pose(1)
cam_proj = vl.get_projection_matrix()

fig, ax = plt.subplots(1)
image_m = show_marker_pose(image, T_cam_marker, cam_proj)
line = ax.imshow(np.asarray(image_m))
ax.set_axis_off()


def update(w):
    image, depth, pose = vl.get_rgbdp(w)
    try:
        T_cam_marker = vl.get_cam_pose(w)
    except FileNotFoundError:
        print("No pose estimation.")
        line.set_data(np.asarray(image))
        return
    image_m = show_marker_pose(image, T_cam_marker, cam_proj)
    line.set_data(np.asarray(image_m))
    fig.canvas.draw_idle()


ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03])  # Position of the slider
slider = Slider(ax_slider, 'Index', 0, len(vl) - 1, valinit=1, valstep=1)

slider.on_changed(update)

plt.show()


def get_tcp_marker_lists(m_dir="pose_marker_one"):
    T_robot_tcp_list = []
    T_cam_marker_list = []
    for i in range(len(vl)):
        try:
            robot_pose = vl.get_robot_pose(i)
            cam_pose = vl.get_cam_pose(i, marker_dir=m_dir)
        except (FileNotFoundError, ValueError):
            continue
        T_robot_tcp_list.append(robot_pose)
        T_cam_marker_list.append(cam_pose)

    return np.array(T_robot_tcp_list), np.array(T_cam_marker_list)


# draw the markers in 3d
import open3d as o3d

plot_o3d = True
# plot_o3d = False
if plot_o3d:
    T_robot_tcp_list, T_cam_marker_list = get_tcp_marker_lists()
    # create one big frame to show the absolute zero point of the plot
    mesh_frames = []
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    mesh_frames.append(mesh_frame)

    for T_robot_tcp, T_cam_marker in zip(T_robot_tcp_list, T_cam_marker_list):
        # create coordinate frames around the zero point and transform them
        # to visualize the transforms

        # # T_robot_tcp shows where the tool is relative to the robot base (small arrows)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        mesh_frame.transform(T_robot_tcp)
        mesh_frames.append(mesh_frame)

        # # you can also view the robot base relative to the tool
        # T_tcp_robot = np.linalg.inv(T_robot_tcp)
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # mesh_frame.transform(T_tcp_robot)
        # mesh_frames.append(mesh_frame)

        # # shows where the cam is relative to the marker (big arrows)
        T_marker_cam = np.linalg.inv(T_cam_marker)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        mesh_frame.transform(T_marker_cam)
        mesh_frames.append(mesh_frame)

        # # shows where the marker is relative to the cam
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.20)
        # mesh_frame.transform(T_cam_marker)
        # mesh_frames.append(mesh_frame)

    # Draw the geometries
    o3d.visualization.draw_geometries(mesh_frames)

# solve 
def vec_to_matrix(x):
    """
    Args:
        x: is our optimization target, a vector of shape (9,).
            [0:3] position transform of T_cam_tcp
            [3:6] rotation transform of T_cam_tcp
                (angles to rotate around xyz axes)
            [6:9] position transform of T_robot_marker (discarded here)

    Returns:
        transformation matrix T_cam_tcp shape (4, 4)

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
    """
    mat = np.eye(4)
    mat[:3, 3] = x[0:3]
    mat[:3, :3] = Rotation.from_euler("xyz", x[3:6]).as_matrix()
    return mat


def pprint(arr):
    return np.array2string(arr.round(5), separator=", ")


def matrix_to_pos_orn(mat):
    """
    Args:
        mat: 4x4 homogeneous transformation

    Returns:
        position: np.array of shape (3,),
        orientation: np.array of shape (4,) -> quaternion xyzw

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html
    """
    orn = Rotation.from_matrix(mat[:3, :3]).as_quat()
    pos = mat[:3, 3]
    return pos, orn


def pose_error(T_tcp_cam, T_robot_tcp_list, T_cam_marker_list):
    """
    returns position error tuple for each entry

    Explanation: We know that T_robot_marker is constant.
    Given our calibrated T_tcp_cam we calculate T_robot_marker for each pair
    of robot position T_robot_tcp and marker detection result T_cam_marker.
    Now assuming perfect marker detection, robot position and calibration,
    the calculated T_robot_marker should be the same for all pairs.
    Therefore we can use derivations of the mean pose as pose error.

    Args:
        T_tcp_cam: Transform from camera to tool center shape (4, 4)
        T_robot_tcp_list: List of robot poses i.e. transform from tool center
            point to robot base, shape (N, 4, 4)
        T_cam_marker_list: List of marker measurements i.e. transform from
            marker to camera, shape (N, 4, 4)
    Returns:
        error: cartesian error (N, 3)
    """
    T_robot_marker_list = []
    for T_robot_tcp, T_cam_marker in zip(T_robot_tcp_list, T_cam_marker_list):
        T_robot_marker = T_robot_tcp @ T_tcp_cam @ T_cam_marker
        T_robot_marker_list.append(T_robot_marker)

    poses = np.array(T_robot_marker_list)[:, :3, 3]
    err = poses - np.mean(poses, axis=0)
    return err

from scipy.optimize import least_squares, minimize


def compute_residuals_gripper_cam(x, T_robot_tcp_list, T_cam_marker_list):
    """
    Calculate predicted positional transform from marker to camera using
    T_cam_tcp, T_robot_marker and T_tcp_robot.
    Compare these predicted values with the observed positional transform
    T_cam_marker to calculate and returns the residuals

    Args:
        x: is our optimization target, a vector of shape (9,).
            [0:3] position transform of T_cam_tcp
            [3:6] rotation transform of T_cam_tcp
                (angles to rotate around xyz axes)
            [6:9] position transform of T_robot_marker
        T_robot_tcp_list: List of robot poses i.e. transform from tool center
            point to robot base, each entry shape (4, 4)
        T_cam_marker_list: List of marker measurements i.e. transform from
            marker to camera, each entry shape (4, 4)

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    """
    T_robot_marker = np.array([*x[6:], 1])  # shape (4, )
    T_cam_tcp = vec_to_matrix(x)  # shape (4, 4)

    # START TODO #################
    residuals = []
    T_tcp_cam = np.linalg.inv(T_cam_tcp)

    for T_robot_tcp, T_cam_marker in zip(T_robot_tcp_list, T_cam_marker_list):
        T_robot_marker_pred = T_robot_tcp @ T_tcp_cam @ T_cam_marker

        residuals.append(T_robot_marker_pred[:3, 3] - T_robot_marker[:3])

    residuals = np.array(residuals).flatten()
    # END TODO ###################

    # len(residuals) will be 144 = 48 samples * 3 (x,y,z), for least-squares optimization
    return residuals


def calibrate_gripper_cam_ls(T_robot_tcp_list, T_cam_marker_list):
    # use scipy least squares to optimize the above function and return the calibration
    # START TODO #################
    x0 = np.zeros(9)

    result = least_squares(
        compute_residuals_gripper_cam,
        x0,
        args=(T_robot_tcp_list, T_cam_marker_list),
    )

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    T_tcp_cam = vec_to_matrix(result.x)
    # END TODO ###################

    assert T_tcp_cam.shape == (4, 4)
    return T_tcp_cam


T_robot_tcp_list, T_cam_marker_list = get_tcp_marker_lists()
T_tcp_cam = calibrate_gripper_cam_ls(T_robot_tcp_list, T_cam_marker_list)
err_ls = pose_error(T_tcp_cam, T_robot_tcp_list, T_cam_marker_list)
err_ls_s = np.sum(err_ls**2, axis=1)
# analyze median error since that is more robust to outliers than mean error
print("median error ls", np.median(err_ls_s))

def calculate_error(T_tcp_cam, T_robot_tcp_list, T_cam_marker_list, inliers=None):
    """
    Args:
        T_tcp_cam: Transform from camera to tool center shape (4,4)
        T_robot_tcp_list: List of robot poses i.e. transform from tool center
            point to robot base, shape (N, 4, 4)
        T_cam_marker_list: List of marker measurements i.e. transform from
            marker to camera, shape (N, 4, 4)
    Returns:
        scalar error for each entry
    """
    T_robot_marker_list = []
    for T_robot_tcp, T_cam_marker in zip(T_robot_tcp_list, T_cam_marker_list):
        T_robot_marker = T_robot_tcp @ T_tcp_cam @ T_cam_marker
        T_robot_marker_list.append(T_robot_marker)

    poses = np.array(T_robot_marker_list)[:, :3, 3]
    if inliers is None:
        mean_pose = np.mean(poses, axis=0)
    else:
        mean_pose = np.mean(poses[inliers], axis=0)
    err = np.sum((poses - mean_pose) ** 2, axis=1)
    return err


import scipy


def calibrate_gripper_cam_de(T_robot_tcp_list, T_cam_marker_list):
    # optimize using least squares as before
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, -0.1])
    result = least_squares(
        fun=compute_residuals_gripper_cam,
        x0=x0,
        method="lm",
        args=(T_robot_tcp_list, T_cam_marker_list),
    )

    x0 = result.x
    bounds = [(x - 0.0001, x + 0.0001) for x in x0]

    def func(x, *args):
        T_cam_tcp = vec_to_matrix(x)
        return calculate_error(T_cam_tcp, *args).mean()

    result2 = scipy.optimize.differential_evolution(
        func=func,
        bounds=bounds,
        args=(T_robot_tcp_list, T_cam_marker_list),
        tol=1e-11,
    )
    T_tcp_cam = np.linalg.inv(vec_to_matrix(result2.x))
    # print(result.x)
    # print(bounds)
    return T_tcp_cam


T_tcp_cam = calibrate_gripper_cam_de(T_robot_tcp_list, T_cam_marker_list)
err_ls = pose_error(T_tcp_cam, T_robot_tcp_list, T_cam_marker_list)
err_ls_s = np.sum(err_ls**2, axis=1)
print("median error ls", np.median(err_ls_s))
