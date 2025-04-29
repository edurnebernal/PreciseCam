#######################################################################
# Adapted code from https://github.com/jinlinyi/PerspectiveFields
#######################################################################
import torch 
import numpy as np
from equilib import grid_sample
from sklearn.preprocessing import normalize
from perspective_fields.visualizer import VisualizerPerspective
from numpy.lib.scimath import sqrt as csqrt

def deg2rad(deg):
    """convert degrees to radians"""
    return deg * np.pi / 180

def diskradius(xi, f):  # compute the disk radius when the image is catadioptric
    return np.sqrt(-(f * f) / (1 - xi * xi))

def crop_distortion(image360_path, f, xi, H, W, az, el, roll):
        """
        Reference: https://github.com/dompm/spherical-distortion-dataset/blob/main/spherical_distortion/spherical_distortion.py
        Crop distorted image with specified camera parameters

        Args:
            image360_path (str): path to image which to crop from
            f (float): focal_length of cropped image
            xi:
            H (int): height of cropped image
            W (int): width of cropped image
            az: camera rotation about camera frame y-axis of cropped image (degrees)
            el: camera rotation about camera frame x-axis of cropped image (degrees)
            roll: camera rotation about camera frame z-axis of cropped image (degrees)
        Returns:
            im (np.ndarray): cropped, distorted image
        """

        u0 = W / 2.0
        v0 = H / 2.0

        grid_x, grid_y = np.meshgrid(list(range(W)), list(range(H)))

        image360 = image360_path.copy()

        ImPano_W = np.shape(image360)[1]
        ImPano_H = np.shape(image360)[0]
        x_ref = 1
        y_ref = 1

        fmin = minfocal(
            u0, v0, xi, x_ref, y_ref
        )  # compute minimal focal length for the image to ve catadioptric with given xi

        # 1. Projection on the camera plane

        X_Cam = np.divide(grid_x - u0, f)
        Y_Cam = -np.divide(grid_y - v0, f)

        # 2. Projection on the sphere

        AuxVal = np.multiply(X_Cam, X_Cam) + np.multiply(Y_Cam, Y_Cam)

        alpha_cam = np.real(xi + csqrt(1 + np.multiply((1 - xi * xi), AuxVal)))

        alpha_div = AuxVal + 1

        alpha_cam_div = np.divide(alpha_cam, alpha_div)

        X_Sph = np.multiply(X_Cam, alpha_cam_div)
        Y_Sph = np.multiply(Y_Cam, alpha_cam_div)
        Z_Sph = alpha_cam_div - xi

        # 3. Rotation of the sphere
        coords = np.vstack((X_Sph.ravel(), Y_Sph.ravel(), Z_Sph.ravel()))
        rot_el = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                np.cos(deg2rad(el)),
                -np.sin(deg2rad(el)),
                0.0,
                np.sin(deg2rad(el)),
                np.cos(deg2rad(el)),
            ]
        ).reshape((3, 3))
        rot_az = np.array(
            [
                np.cos(deg2rad(az)),
                0.0,
                -np.sin(deg2rad(az)),
                0.0,
                1.0,
                0.0,
                np.sin(deg2rad(az)),
                0.0,
                np.cos(deg2rad(az)),
            ]
        ).reshape((3, 3))
        rot_roll = np.array(
            [
                np.cos(deg2rad(roll)),
                np.sin(deg2rad(roll)),
                0.0,
                -np.sin(deg2rad(roll)),
                np.cos(deg2rad(roll)),
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ).reshape((3, 3))
        sph = rot_el.dot(rot_roll.T.dot(coords))
        sph = rot_az.dot(sph)

        sph = sph.reshape((3, H, W)).transpose((1, 2, 0))
        X_Sph, Y_Sph, Z_Sph = sph[:, :, 0], sph[:, :, 1], sph[:, :, 2]

        # 4. cart 2 sph
        ntheta = np.arctan2(X_Sph, Z_Sph)
        nphi = np.arctan2(Y_Sph, np.sqrt(Z_Sph**2 + X_Sph**2))

        pi = np.pi

        # 5. Sphere to pano
        min_theta = -pi
        max_theta = pi
        min_phi = -pi / 2.0
        max_phi = pi / 2.0

        min_x = 0
        max_x = ImPano_W - 1.0
        min_y = 0
        max_y = ImPano_H - 1.0

        ## for x
        a = (max_theta - min_theta) / (max_x - min_x)
        b = max_theta - a * max_x  # from y=ax+b %% -a;
        nx = (1.0 / a) * (ntheta - b)

        ## for y
        a = (min_phi - max_phi) / (max_y - min_y)
        b = max_phi - a * min_y  # from y=ax+b %% -a;
        ny = (1.0 / a) * (nphi - b)
        lat = nphi.copy()
        xy_map = np.stack((nx, ny)).transpose(1, 2, 0)

        # 6. Final step interpolation and mapping
        # im = np.array(my_interpol.interp2linear(image360, nx, ny), dtype=np.uint8)
        im = grid_sample.numpy_grid_sample.default(
            image360.transpose(2, 0, 1), np.stack((ny, nx))
        ).transpose(1, 2, 0)
        if (
            f < fmin
        ):  # if it is a catadioptric image, apply mask and a disk in the middle
            r = diskradius(xi, f)
            DIM = im.shape
            ci = (np.round(DIM[0] / 2), np.round(DIM[1] / 2))
            xx, yy = np.meshgrid(
                list(range(DIM[0])) - ci[0], list(range(DIM[1])) - ci[1]
            )
            mask = np.double((np.multiply(xx, xx) + np.multiply(yy, yy)) < r * r)
            mask_3channel = np.stack([mask, mask, mask], axis=-1).transpose((1, 0, 2))
            im = np.array(np.multiply(im, mask_3channel), dtype=np.uint8)

        col = nphi[:, W // 2]
        zero_crossings_rows = np.where(np.diff(np.sign(col)))[0]
        if len(zero_crossings_rows) >= 2:
            print("WARNING | Number of zero crossings:", len(zero_crossings_rows))
            zero_crossings_rows = [zero_crossings_rows[0]]

        if len(zero_crossings_rows) == 0:
            offset = np.nan
        else:
            assert col[zero_crossings_rows[0]] >= 0
            assert col[zero_crossings_rows[0] + 1] <= 0
            dy = col[zero_crossings_rows[0] + 1] - col[zero_crossings_rows[0]]
            offset = zero_crossings_rows[0] - col[zero_crossings_rows[0]] / dy
            assert col[zero_crossings_rows[0]] / dy <= 1.0
        # Reproject [nx, ny+epsilon] back
        epsilon = 1e-5
        end_vector_x = nx.copy()
        end_vector_y = ny.copy() - epsilon
        # -5. pano to Sphere
        a = (max_theta - min_theta) / (max_x - min_x)
        b = max_theta - a * max_x  # from y=ax+b %% -a;
        ntheta_end = end_vector_x * a + b
        ## for y
        a = (min_phi - max_phi) / (max_y - min_y)
        b = max_phi - a * min_y
        nphi_end = end_vector_y * a + b
        # -4. sph 2 cart
        Y_Sph = np.sin(nphi)
        X_Sph = np.cos(nphi_end) * np.sin(ntheta_end)
        Z_Sph = np.cos(nphi_end) * np.cos(ntheta_end)
        # -3. Reverse Rotation of the sphere
        coords = np.vstack((X_Sph.ravel(), Y_Sph.ravel(), Z_Sph.ravel()))
        sph = rot_roll.dot(rot_el.T.dot(rot_az.T.dot(coords)))
        sph = sph.reshape((3, H, W)).transpose((1, 2, 0))
        X_Sph, Y_Sph, Z_Sph = sph[:, :, 0], sph[:, :, 1], sph[:, :, 2]

        # -1. Projection on the image plane

        X_Cam = X_Sph * f / (xi * csqrt(X_Sph**2 + Y_Sph**2 + Z_Sph**2) + Z_Sph) + u0
        Y_Cam = -Y_Sph * f / (xi * csqrt(X_Sph**2 + Y_Sph**2 + Z_Sph**2) + Z_Sph) + v0
        up = np.stack((X_Cam - grid_x, Y_Cam - grid_y)).transpose(1, 2, 0)
        up = normalize(up.reshape(-1, 2)).reshape(up.shape)

        return im, ntheta, nphi, offset, up, lat, xy_map

def minfocal(u0, v0, xi, xref=1, yref=1):
    """compute the minimum focal for the image to be catadioptric given xi"""
    value = -(1 - xi * xi) * ((xref - u0) * (xref - u0) + (yref - v0) * (yref - v0))

    if value < 0:
        return 0
    else:
        return np.sqrt(value) * 1.0001

#------------------------------------------------

def draw_perspective_fields(
    img_rgb, up, latimap, color=None, density=10, arrow_inv_len=20, return_img=True
):
    """draw perspective field on top of input image

    Args:
        img_rgb (np.ndarray): input image
        up (np.ndarray): gravity field (h, w, 2)
        latimap (np.ndarray): latitude map (h, w) (radians)
        color ((float, float, float), optional): RGB color for up vectors. [0, 1]
                                                 Defaults to None.
        density (int, optional): Value to control density of up vectors.
                                 Each row has (width // density) vectors.
                                 Each column has (height // density) vectors.
                                 Defaults to 10.
        arrow_inv_len (int, optional): Value to control vector length
                                       Vector length set to (image plane diagonal // arrow_inv_len).
                                       Defaults to 20.
        return_img (bool, optional): bool to control if to return np array or VisImage

    Returns:
        image blended with perspective fields.
    """
    visualizer = VisualizerPerspective(img_rgb.copy())
    vis_output = visualizer.draw_lati(latimap)
    if torch.is_tensor(up):
        up = up.numpy().transpose(1, 2, 0)
    im_h, im_w, _ = img_rgb.shape
    x, y = np.meshgrid(
        np.arange(0, im_w, im_w // density), np.arange(0, im_h, im_h // density)
    )
    x, y = x.ravel(), y.ravel()
    arrow_len = np.sqrt(im_w**2 + im_h**2) // arrow_inv_len
    end = up[y, x, :] * arrow_len
    if color is None:
        color = (0, 1, 0)
    vis_output = visualizer.draw_arrow(x, y, end[:, 0], -end[:, 1], color=color)
    if return_img:
        return vis_output.get_image()
    else:
        return vis_output
