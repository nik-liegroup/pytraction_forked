from scipy.sparse import spdiags
import numpy as np


gamma_1 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]).flatten()
gamma_2 = np.array([[111,222,333,444], [555,666,777,888], [999,101010,111111,121212], [131313,141414,151515,161616]]).flatten()
gamma_3 = gamma_2
gamma_4 = np.array([[17,18,19,20], [21,22,23,24], [25,26,27,28], [29,30,31,32]]).flatten()

pad = np.zeros(gamma_2.shape)

diagonals = np.array([np.concatenate([gamma_1, gamma_4]).flatten(),
                          np.concatenate([pad, gamma_2]).flatten(),
                          np.concatenate([gamma_3, pad]).flatten()])

gamma_glob = spdiags(data=diagonals,
                     diags=(0, 4 * 4, -4 * 4),
                     m=2*4*4,
                     n=2*4*4).toarray()

print(gamma_glob)
