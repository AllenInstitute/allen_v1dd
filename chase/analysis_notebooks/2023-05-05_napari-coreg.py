import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy.ndimage as ndi
from tifffile import tifffile
import napari

from allen_v1dd.client import OPhysClient

EM_VOLUME_FILENAME = "/Users/chase/Desktop/V1DD_coregistration/v1dd_mip7_calib-1.tif"
FIJI_TRANSPOSE_COLUMNS = [2, 1, 0] # convert [x', y', z'] (Fiji) to [z, y, x] (tiff)
# FIJI_TRANSPOSE_COLUMNS = [1, 0, 2] # convert [x', y', z'] (Fiji) to [y, x, z] (Napari)
FIJI_VOXEL_SIZE_MICRONS = np.array([0.620800093368334, 0.620800093368334, 0.18]) # From Fiji calibration XML file; dims [x', y', z']

em_images = tifffile.imread(EM_VOLUME_FILENAME) # Dimensions: [Y, X, Z]
EM_IMAGES_ORIG_SHAPE = em_images.shape
print("EM image shape:", EM_IMAGES_ORIG_SHAPE)
# em_images = em_images.transpose(2, 0, 1) # New dimensions: [X, Y, Z]
# print("New shape:", em_images.shape)

FIJI_TO_TIFF_Z_SCALE = (EM_IMAGES_ORIG_SHAPE[0] / 4416) / FIJI_VOXEL_SIZE_MICRONS[2]

def convert_fiji_to_napari(fiji_coords):
    """
    Fiji coords are in [x', y', z']; Napari coords are in [z, y, x]
    (Where +x is right, +y is down, and +z is inward (further depths).)

    Fiji image is scaled to microns and reshaped to size [2720, 1963, 4416]

    Steps to transform:
        1. Convert fiji_coords to fiji_voxels by dividing by microns resolution
        2. Transpose fiji_voxels from [x', y', z'] to [z', y', x'], to align with tiff coords
        3. Rescale fiji_voxels to tiff_voxels by 

    Args:
        fiji_coords (array_like): _description_
    """
    orig_shape = fiji_coords.shape
    fiji_coords = np.atleast_2d(fiji_coords)    
    fiji_voxels = fiji_coords# / FIJI_VOXEL_SIZE_MICRONS

    # Setup transform
    fiji_voxels_transposed = fiji_voxels[:, FIJI_TRANSPOSE_COLUMNS]
    real_image_size = np.array([2720, 1963, 4416])[FIJI_TRANSPOSE_COLUMNS]
    
    napari_coords = fiji_voxels_transposed# * EM_IMAGES_ORIG_SHAPE / real_image_size

    napari_coords[:, 0] *= FIJI_TO_TIFF_Z_SCALE

    return napari_coords.reshape(orig_shape)
    

# transform_matrix = np.diag(voxel_size_microns)

# Load napari viewer
viewer = napari.Viewer(title="V1DD EM Coregistration")

# real_image_size = np.array([2720, 1963, 4416])[FIJI_TRANSPOSE_COLUMNS]
# real_image_size_scaling = real_image_size / EM_IMAGES_ORIG_SHAPE
# tiff_to_microns = np.diag([FIJI_VOXEL_SIZE_MICRONS[1] * real_image_size_scaling[1], FIJI_VOXEL_SIZE_MICRONS[0] * real_image_size_scaling[2], 0])
# viewer.add_image(em_images, colormap="gray_r", opacity=1, name="EM Volume", affine=tiff_to_microns)


XYZ_TO_ZYX = [2, 1, 0]
ZYX_TO_XYZ = [2, 1, 0] # note the transform is the same as above but including this for some other methods
TIFF_SHAPE_ZYX = np.array(em_images.shape) # tiff file has axes (Z, Y, X)
TIFF_SHAPE_XYZ = TIFF_SHAPE_ZYX[XYZ_TO_ZYX]
TIFF_SIZE_MICRONS_XYZ = np.array([2720, 1963, 4416]) * [0.620800093368334, 0.620800093368334, 0.18] # voxels * (microns / voxel) = microns
TIFF_SCALE_ZYX = TIFF_SIZE_MICRONS_XYZ[XYZ_TO_ZYX] / TIFF_SHAPE_ZYX

# Transform tiff coords --> microns: * TIFF_SCALE
# Transform microns --> tiff cords: / TIFF_SCALE



viewer.add_image(em_images, colormap="gray_r", opacity=1, name="EM Volume", scale=TIFF_SCALE_ZYX)



viewer.dims.order = (1, 0, 2) # Scroll through y (depth)
viewer.dims.axis_labels = ("X", "Y", "Z")

ophys_client = OPhysClient("chase_local")
session = ophys_client.load_ophys_session("M409828_13")


napari_state = {
    "current_plane": None,
    "current_roi": None,
    "is_valid": None
}


# Add an image layer for plane 0 max proj
plane = 0
proj_raw_mean, proj_raw_max, proj_de_mean, proj_de_max, proj_de_corr = session.get_plane_projection_images(plane)


roi_df = pd.read_csv("/Users/chase/Desktop/v1dd_M409828_13_roi_depths.csv")


def get_current_roi_image_and_center():
    plane = napari_state["current_plane"]
    current_roi = napari_state["current_roi"]
    colored_mask = None
    roi_center = None

    for roi in roi_df.roi.values[roi_df.plane == plane]:
        roi_mask = session.get_roi_image_mask(plane, roi)

        if colored_mask is None:
            colored_mask = np.full((*roi_mask.shape, 4), np.nan)
        
        if roi == current_roi:
            color = "red"
            roi_center = np.mean(np.where(roi_mask), axis=1)
        else:
            color = "magenta"

        color = mpl.colors.to_rgb(color)
        already_has_roi = np.all(~np.isnan(colored_mask), axis=2)
        colored_mask[roi_mask & already_has_roi] = [*color, 0.8] # high alpha for overlapping ROIs
        colored_mask[roi_mask & ~already_has_roi] = [*color, 0.5]
    
    return colored_mask, roi_center


OPHYS_MICRONS_PER_PIXEL = 400 / 512
ROI_MASK_LAYER_NAME = "OPhys ROI Masks"

# viewer.add_image(proj_de_max, colormap="inferno", name=f"Plane {plane} Decorrelated Max", affine=ophys_to_microns_transform, visible=False)

def get_layer_by_name(layer_name):
    for layer in viewer.layers:
        if layer.name == layer_name:
            return layer
    return None

def set_current_roi(plane, roi):
    napari_state["current_plane"] = plane
    napari_state["current_roi"] = roi

    colored_mask, roi_center = get_current_roi_image_and_center()

    ophys_to_microns_transform = np.array([
        [OPHYS_MICRONS_PER_PIXEL, 0, 542 - roi_center[0]*OPHYS_MICRONS_PER_PIXEL],
        [0, OPHYS_MICRONS_PER_PIXEL, 686 - roi_center[1]*OPHYS_MICRONS_PER_PIXEL],
        [0, 0, 0]
    ])

    roi_masks_layer = get_layer_by_name(ROI_MASK_LAYER_NAME)

    if roi_masks_layer is None:
        roi_masks_layer = viewer.add_image(
            colored_mask,
            interpolation2d="nearest",
            name=ROI_MASK_LAYER_NAME,
            affine=ophys_to_microns_transform
        )
    else:
        roi_masks_layer.data = colored_mask
        # roi_masks_layer.affine = ophys_to_microns_transform


    # viewer.dims.range = ((240, 300, 1), viewer.dims.range[1], viewer.dims.range[2])
    # viewer.dims.set_point(axis=0, value=coregistration_points[0, 0])
    update_hud()


coreg_cells = { # roi: [x, y, z]
    (0, 117): [686.845, 542.305, 193.935],
    (0, 26): [747, 564, 208]
}

coreg_cells_rois = []
coregistration_points = []

for roi, pos in coreg_cells.items():
    coreg_cells_rois.append(roi)
    coregistration_points.append(pos)

coregistration_points = convert_fiji_to_napari(np.array(coregistration_points))

print(coregistration_points)

viewer.add_points(
    data=coregistration_points[:, 1:],
    features=dict(
        ophys_plane=[x[0] for x in coreg_cells_rois],
        ophys_roi=[x[1] for x in coreg_cells_rois]
    ),
    text = dict(
        string = "{ophys_plane}-{ophys_roi}",
        size = 6,
        color = "magenta",
        translation = [-5, 0]
    )
)

def get_cursor_xyz_microns():
    # cursor in ZYX coords with units microns
    return np.array(viewer.cursor.position)[ZYX_TO_XYZ]


def update_hud():
    text = viewer.text_overlay

    text.visible = True
    text.color = "white"
    # text.blending = napari.layers.base._base_constants.Blending.OPAQUE

    roi = f"{napari_state['current_plane']}-{napari_state['current_roi']}"
    is_valid = napari_state["is_valid"]
    cursor_coords = get_cursor_xyz_microns()

    lines = [
        f"Current ROI: {roi}",
        f"Is valid [v]: {'N/A' if is_valid is None else 'YES' if is_valid else 'NO'}",
        "",
        f"Viewer depth: {cursor_coords[1]:.0f} µm",
        f"Pointer coords: ({', '.join(f'{x:.0f}' for x in cursor_coords)}) µm^3"
        "",
        "Next ROI [n]"
    ]

    text.text = "\n".join(lines)

TRANSFORM_COREG_TO_CAVE = np.load("/Users/chase/Desktop/em_coreg2final_tform.npy")
TRANSFORM_CAVE_TO_COREG = np.linalg.inv(TRANSFORM_COREG_TO_CAVE)

def get_caveclient_nm(coreg_pos):
    orig_shape = np.shape(coreg_pos) # save original state
    coreg_pos = np.atleast_2d(coreg_pos) # make 2d
    coreg_pos = np.hstack([coreg_pos, np.ones((len(coreg_pos), 1))]) # append column of ones
    nm = TRANSFORM_COREG_TO_CAVE.dot(coreg_pos.T).T # coreg_pos.T because columns are multiplied
    nm = nm[:, :3] * [4.85, 4.85, 45] # drop last column of ones; convert to nm by multiplying by voxel resolution of transform
    return nm.reshape(orig_shape)

update_hud()

@viewer.bind_key("v")
def mark_valid(viewer):
    # Toggle validity state of current ROI
    napari_state["is_valid"] = napari_state["is_valid"] in (None, False)
    update_hud()
    # TODO: Save to file, etc.

@viewer.bind_key("p")
def print_ngl_coords(viewer):
    cursor_coords = get_cursor_xyz_microns()
    caveclient_nm = get_caveclient_nm(cursor_coords)
    ngl_vx = caveclient_nm / [9, 9, 45]

    print()
    print("NGL coords [9, 9, 45]:")
    print(", ".join(f"{x:.0f}" for x in ngl_vx))
    print()


def cycle_roi(offset):
    plane, roi = napari_state["current_plane"], napari_state["current_roi"]
    all_rois_in_plane = list(roi_df.roi.values[roi_df.plane == plane])
    index = all_rois_in_plane.index(roi)
    new_roi = all_rois_in_plane[index+offset]
    set_current_roi(plane, new_roi)

@viewer.bind_key("r")
def next_roi(viewer):
    cycle_roi(+1)

@viewer.bind_key("Shift-r")
def prev_roi(viewer):
    cycle_roi(-1)


def on_mouse_click(viewer, event):
    print("Double click", viewer.cursor.position)

    set_current_roi(0, 26)

    # roi_masks_layer = get_layer_by_name(ROI_MASK_LAYER_NAME)
    # print(dir(roi_masks_layer.affine))

set_current_roi(*coreg_cells_rois[0])

viewer.mouse_double_click_callbacks.append(on_mouse_click)

# Update Z text
viewer.dims.events.current_step.connect(lambda event: update_hud())

viewer.dims.axis_labels = ("Z", "Y", "X")

viewer.scale_bar.visible = True
viewer.scale_bar.unit = "µm"

if __name__ == "__main__":
    # napari.settings.get_settings().appearance.theme = "system"
    napari.settings.get_settings().appearance.theme = "light"
    # napari.settings.get_settings().appearance.theme = "dark"
    napari.run()