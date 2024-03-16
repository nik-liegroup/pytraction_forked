from pytraction.process import (calculate_traction_map, compute_piv,
                                iterative_piv, align_slice)
import numpy as np

shift = 10
img = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
ref = np.roll(img, shift, axis=1)
ref[:, :shift] = 0

align_slice(img, ref)
