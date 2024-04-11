import os
from pytraction import (TractionForceConfig, process_stack, plot, TractionForceDataset)


def t_fourier_xu():
    analyse_folder = os.path.join('..', 'data', 'example4')
    E = 10000
    pix_per_mu = 9.64

    config_path = os.path.join(analyse_folder, 'config.yaml')

    img_path = os.path.join(analyse_folder,
                            '2DTFM_10kPa_hPAAGel_2_3T3Fibroblasts_2023_12_14_TimePos_Series_Position 7.tiff')
    ref_path = os.path.join(analyse_folder,
                            '2DTFM_10kPa_hPAAGel_2_3T3Fibroblasts_2023_12_14_Pos_Series_Reference_Position 7.tiff')
    roi_path = os.path.join(analyse_folder,
                            '2DTFM_10kPa_hPAAGel_2_3T3Fibroblasts_2023_12_14_TimePos_Series_Position 7.roi')
    data_path = os.path.join(analyse_folder, 'h5',
                             '2DTFM_10kPa_hPAAGel_2_3T3Fibroblasts_2023_12_14_TimePos_Series_Position 7.h5')
    png_path = os.path.join(analyse_folder, 'png',
                            '2DTFM_10kPa_hPAAGel_2_3T3Fibroblasts_2023_12_14_TimePos_Series_Position 7.png')

    traction_config = TractionForceConfig(E=E,
                                          scaling_factor=pix_per_mu,
                                          window_size=64,
                                          config_path=config_path,
                                          segment=False)

    img, ref, roi = traction_config.load_data(img_path=img_path, ref_path=ref_path, roi_path=roi_path)

    log = process_stack(img[:1, :, :, :],
                        ref,
                        traction_config,
                        roi=roi,
                        crop=True,
                        bead_channel=1,
                        cell_channel=0)
    log.save(data_path)
    fig, ax = plot(log, frame=0, mask=True, figsize=(20, 20))
    ax[1].remove()
    fig.savefig(png_path, dpi=fig.dpi, bbox_inches="tight")


t_fourier_xu()
