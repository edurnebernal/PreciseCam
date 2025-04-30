import numpy as np
import perspective_fields.pano_utils as pf


def sample_parameters():
    """
    Sample camera parameters for an equiangular panorama image.
    The function generates a list of sampled parameters (roll, yaw, pitch, vertical field of view (vfov), and xi)
    for different camera orientations.

    arguments:
        None
    returns:
        sampled_parameters: list of sampled camera parameters
    """
    sampled_parameters = []

    # Sample two yaw values sufficiently separated to avoid overlap
    first_yaw = np.random.randint(0, 180)

    if first_yaw > 100:
        second_yaw = np.random.randint(first_yaw + 80, 360)
    elif first_yaw < 80:
        second_yaw = np.random.randint(180, 360 + first_yaw - 80)
    else:
        second_yaw = np.random.randint(180, 360)

    yaws = np.array([first_yaw, second_yaw], dtype=float)

    # Sample three pitches: top, bottom, and equator for each yaw (image y-axis)
    for yaw in yaws:
        top_pitch = np.random.randint(1, 89)
        bottom_pitch = (
            np.random.randint(-89, -1) if top_pitch > 30 else np.random.randint(-89, 1)
        )
        equator_pitch = np.random.randint(-20, 20)

        pitches = np.array([top_pitch, bottom_pitch, equator_pitch], dtype=float)

        # xi: low and high range
        xis = np.round(np.random.uniform([0.01, 0.5], [0.5, 0.99]), 2)

        # vfovs: low and high range
        vfovs = [np.random.randint(15, 60), np.random.randint(60, 140)]

        # Random rolls: shape (3 pitches x 2 xis x 2 vfovs)
        rolls = np.random.randint(-89, 89, size=(len(pitches), len(xis), len(vfovs)))

        for i, pitch in enumerate(pitches):
            for j, xi in enumerate(xis):
                for k, vfov in enumerate(vfovs):
                    roll = rolls[i, j, k]
                    sampled_parameters.append(
                        [float(roll), float(yaw), float(pitch), float(vfov), float(xi)]
                    )

    return sampled_parameters


def obtain_pf(panorama, roll, yaw, pitch, vfov, xi, height, width):
    """
    Computes the perspective field for specific camera model using the unified spherical camera model.

    arguments:
        panorama: panorama image (numpy array)
        roll: camera rotation about camera frame z-axis of cropped image (degrees)
        yaw: camera rotation about camera frame y-axis of cropped image (degrees)
        pitch: camera rotation about camera frame x-axis of cropped image (degrees)
        vfov: vertical field of view (degrees)
        xi: xi parameter for the camera model
        height: height of the cropped image
        width: width of the cropped image

    returns:
        image_crop: cropped image (numpy array)
        pf_us: perspective field image (numpy array). Encoded as 3 channels: (u,v) up vector and latitude.
    """
    x = -np.sin(np.radians(vfov / 2))
    z = np.sqrt(1 - x**2)

    image_crop, _, _, _, up, lat, _ = pf.crop_distortion(
        panorama,
        f=-0.5 * (width / 2) * (xi + z) / x,
        xi=xi,
        H=height,
        W=width,
        az=yaw,
        el=-pitch,
        roll=roll,
    )

    # Scale up vector to [0, 255] range and latitude to [0, 255] range
    gravity = (up + 1) * 127.5
    latitude = np.expand_dims((np.degrees(lat) + 90) * (255 / 180), axis=-1)

    # Concatenate gravity and latitude to create the perspective field image
    pf_us = np.concatenate([gravity, latitude], axis=2)

    return image_crop, pf_us
