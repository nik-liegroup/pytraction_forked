import glob
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

from pytraction.utils import allign_slice
from pytraction.piv import PIV
from pytraction.fourier import fourier_xu

from pytraction.utils import allign_slice
from pytraction.piv import PIV
from pytraction.fourier import fourier_xu
from pytraction.reg_fourier import reg_fourier_tfm
from pytraction.optimal_lambda import optimal_lambda


from openpiv import tools, pyprocess, validation, filters, scaling
from openpiv import widim, windef
from mpl_toolkits.axes_grid1 import make_axes_locatable



def clach(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(100,100))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


frame = 0
channel = 0

meshsize = 10 # grid spacing in pix
pix_durch_mu = 1.3
scaling_factor = pix_durch_mu

E = 1100 # Young's modulus in Pa
s = 0.5 # Poisson's ratio

TFM = True

# load data from ryan
files = glob.glob('C:\\Users\\ryan\Google Drive\\TFM Cambridge\\2021\\Frogs\\TFM replicates\\e1_FG2_1_20210305\\DATA\\*.tif')
images = [x for x in files if '_ref' not in x]

for img_stack_name in images:
    
    img1 = io.imread(img_stack_name)
    ref1 = io.imread(img_stack_name.replace('.tif', '_ref.tif'))
    
    print(img1.shape)


    frame_b= np.array(img1[frame, channel, :, :], dtype='uint8')
    frame_a = np.array(ref1[channel,:,:], dtype='uint8')

    nframes = img1.shape[0]


    img_path, name = os.path.split(img_stack_name)
    save_path = os.path.join(img_path, name.split('.')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    

    for frame in range(nframes):
    # for frame in :
        frame_b= np.array(img1[frame, channel, :, :], dtype='uint8')
        frame_a = np.array(ref1[channel,:,:], dtype='uint8')

        frame_a = allign_slice(frame_a, frame_b)

        img = np.stack([frame_b, frame_a])

        # cv2.imwrite(f'{save_path}/axon_{frame}.png', axons_img
        io.imsave(f'{save_path}/axon_test{frame}.tif',img)

        # piv_obj = PIV(window_size=32, search_area_size=32, overlap=16)
        # x, y, u, v = piv_obj.base_piv(frame_a, frame_b)


        settings = windef.Settings()

        settings.correlation_method='linear'

        x,y,u,v, mask = widim.WiDIM(frame_a.astype(np.int32), 
                                    frame_b.astype(np.int32), 
                                    np.ones_like(frame_a).astype(np.int32), 
                                    min_window_size=32, 
                                    overlap_ratio=0.5, 
                                    coarse_factor=0, 
                                    dt=1, 
                                    validation_method='mean_velocity', 
                                    trust_1st_iter=0, 
                                    validation_iter=3, 
                                    tolerance=1.5, 
                                    nb_iter_max=1, 
                                    sig2noise_method='peak2peak')



        u = u*scaling_factor
        v = v*scaling_factor

        if not TFM:
            rgcopy = [x,y,u,v]

            mag = np.sqrt(u**2 + v**2) * scaling_factor
            mag = mag.flatten()

            fig, ax = plt.subplots(1,2)

            ax[0].quiver(x,y,u,v)
            ax[1].hist(mag)
            plt.show()


        if TFM:



            rgcopy = [x,y,u,v]

            # plt.quiver(x,y,u,v)
            # plt.show()

            if frame == 0:
                noise = 10
                xn, yn, un, vn = x[:noise],y[:noise],u[:noise],v[:noise]
                noise_vec = np.array([un.flatten(), vn.flatten()])

                varnoise = np.var(noise_vec)
                beta = 1/varnoise



            pos = np.array([x.flatten(), y.flatten()])
            vec = np.array([u.flatten(), v.flatten()])

            # calculate with E=1 and rescale Young's modulus at the end
            # prepare Fourier-tranformed Green's function and displacement
            # NOTE: fourier_X_u shifts the positions of the displacement
            # vectors such that the tractions are displayed in the deformed frame
            # where the cell is localized
            grid_mat, i_max, j_max, X, fuu, Ftux, Ftuy, u = fourier_xu(pos,vec, meshsize, 1, s,[])

            # get lambda from baysian bad boi 
            L, evidencep, evidence_one = optimal_lambda(beta, fuu, Ftux, Ftuy, 1, s, meshsize, i_max, j_max, X, 1)

            # do the TFM
            pos,traction,traction_magnitude,f_n_m,_,_ = reg_fourier_tfm(Ftux, Ftuy, L, 1, s, meshsize, i_max, j_max, grid_mat, pix_durch_mu, 0)


            #rescale traction with proper Young's modulus
            traction = E*traction
            traction_magnitude = E*traction_magnitude
            f_n_m = E*f_n_m

            #display a heatmap of the spatial traction distribution
            # fnorm = np.sqrt(f_n_m[:,:,1]**2 + f_n_m[:,:,0]**2)**0.5

            print(L)

            # plt.show()



            img = traction_magnitude.reshape(i_max, j_max).T
            # stress_stack.append(img)

            ax = plt.subplot(111)
            x, y, u, v = rgcopy
            img = np.flip(img, axis=0)
            im = ax.imshow(img, interpolation='bicubic', cmap='jet',extent=[x.min(), x.max(), y.min(), y.max()], vmin=0)
            ax.quiver(x, y, u, v)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.savefig(f'{save_path}/{frame}.png')
            plt.close()


print('comleated')
# print(stress_stack)



        # ax = plt.subplot(111)
        # x, y, u, v = rgcopy
        # img = np.flip(img, axis=0)
        # im = ax.imshow(img, interpolation='bicubic', cmap='jet',extent=[x.min(), x.max(), y.min(), y.max()] )
        # ax.quiver(x, y, u, v)

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
        # plt.show()
