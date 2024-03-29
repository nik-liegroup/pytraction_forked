import matplotlib.pyplot as plt
import numpy as np
import imageio
import io

from typing import Tuple, Type, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytraction.tractionforcedataset import TractionForceDataset

def tfm_plot(
        tfm_dataset: type(TractionForceDataset),
        vec_field: str,
        frames: Union[list, int] = 0,
        heat_map: bool = True,
        gif: int = 0,
        mask: bool = True,
        vmax: float = None,
        figsize: tuple = (16, 16)
):
    """
    Plots 2D vector field stored in TractionForceDataset on top of traction field heatmap or cell image.

    @param  tfm_dataset: TractionForceDataset file containing TFM results.
    @param  vct_field: Vector field to be drawn. Options include "piv_disp" (PIV displacement field), "int_disp"
    (interpolated displacement field) and "trct" (traction field).
    @param  frames: List of numbers indidcating which frames are used.
    @param  heat_map: Set heat map of traction magnitude as background (True) or use image of cell (False).
    @param  mask: Wether to use mask in plot or not.
    @param  vmax: Maximum value of heatmap colorbar, not used if heat_map set to False.
    @param  figsize: Size (width, height) of plot in inch.
    """

    # Get full identifier names for corresponding vector field
    if vec_field == "piv_disp":
        vec_field = "deformation"
        pos_field = "position"
        cbar_label = "PIV deformations [µm]"
    elif vec_field == "int_disp":
        vec_field = "deformation_interpolated"
        pos_field = "position_interpolated"
        cbar_label = "Interpolated deformations [µm]"
    elif vec_field == "trct":
        vec_field = "traction"
        pos_field = "position_interpolated"
        cbar_label = "Traction stress [Pa]"
    else:
        msg = (f"{vec_field} is not an option for vct_field, please use one of the following: piv_disp, int_disp or"
               f"trct.")
        raise RuntimeError(msg)

    if isinstance(frames, int):
        frames = np.asarray([frames], dtype=int)

    # Create empty arrays
    pos, vec, vec_dim, corr_stack, cell_img, mask = [], [], [], [], [], []

    # Find the most common image dimension in time-series
    for frame in frames:
        frame = int(frame)
        dim_tmp = (tfm_dataset[frame][vec_field][0]).shape
        # Append dimension of frames vector fild to list
        vec_dim.append(dim_tmp[1:])

    # Get most occuring dimension tuple
    comm_dim = max(set(vec_dim), key=vec_dim.count)

    # Extract time-series results frame-by-frame
    for frame in frames:
        frame = int(frame)
        tfm_frame = tfm_dataset[frame]

        # Filter images with dimensions different from most commmon dimension
        if (tfm_frame[vec_field][0]).shape[1:] != comm_dim:
            continue

        pos.append(tfm_frame[pos_field][0])
        vec.append(tfm_frame[vec_field][0])
        corr_stack.append(tfm_frame["drift_corrected_stack"][0])
        cell_img.append(tfm_frame["cell_image"][0])
        mask.append(tfm_frame["mask_roi"][0])

    # Calculate mean fields from time-series frames
    pos = np.mean(pos, axis=0)
    vec = np.mean(vec, axis=0)
    mask = np.mean(mask, axis=0)
    mask = np.ma.masked_where(mask == 255, mask)

    pos_x, pos_y = pos[:, :, 0], pos[:, :, 1]
    vec_x, vec_y = vec[:, :, 0], vec[:, :, 1]

    # Create plots
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    if heat_map is True:
        # Calculate background traction map
        traction_map = np.sqrt(vec_x ** 2 + vec_y ** 2)
        traction_map = np.flipud(traction_map)
        vmax = np.max(traction_map) if not vmax else vmax

        # Plot background traction map
        main_clr = ax[0].imshow(
            traction_map,
            interpolation="bicubic",
            cmap="jet",
            extent=[pos_x.min(), pos_x.max(), pos_y.min(), pos_y.max()],
            vmin=0,
            vmax=vmax,
        )
        ax[0].set_axis_off()

        # Plot vector field
        ax[0].quiver(pos_x, pos_y, vec_x, vec_y)

    else:
        # Plot background cell image
        ax[0].imshow(
            cell_img[0],
            cmap="gray",
            alpha=0.6,
            extent=[pos_x.min(), pos_x.max(), pos_y.min(), pos_y.max()],
        )
        ax[0].set_axis_off()

        # Plot vector field
        main_clr = ax[0].quiver(pos_x, pos_y, vec_x, vec_y, np.sqrt(vec_x**2 + vec_y**2), cmap='jet')

    # Plot colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(main_clr, cax=cax)
    cbar.set_label(cbar_label, rotation=270, labelpad=35, size=25)
    cbar.ax.tick_params(labelsize=20)
    ax[1].remove()

    return fig, ax


def tfm_gif(figs: list, sys_path: str, fps: float = 5):
    """
    Creates .gif movie from matplotlib figures with provided frames per second.
    """
    # Set colorbar maximum of all figures equally to the largest
    figs, vmax = set_cbar_max(figs)

    images = []
    for fig in figs:
        # Save images to buffer as .png
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=fig.dpi, bbox_inches="tight")

        # Rewind the pointer to beginning of file and read
        buffer.seek(0)
        image = imageio.imread(buffer)
        images.append(image)

    # Save the images to a GIF file
    imageio.mimsave(sys_path, images, duration=1/fps)



def set_cbar_max(figs: list):
    """
    Get maximum colorbar value of figures in list and applies it to all.
    """
    max_vmax = 0
    for fig in figs:
        cbar = fig.axes[0].collections[0].colorbar
        vmax = cbar.vmax
        max_vmax = max(max_vmax, vmax)

    for fig in figs:
        cbar = fig.axes[0].collections[0].colorbar
        cbar.mappable.set_clim(vmin=cbar.vmin, vmax=vmax)

    return figs, vmax
