import tensorflow as tf
import keras
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from neuron.layers import SpatialTransformer
from pathlib import Path

tf.enable_eager_execution()

rd = Path(__file__)
im = sitk.ReadImage(str(rd.parent / "MRHead.nrrd"))
im_arr = sitk.GetArrayFromImage(im)
im_arr = np.swapaxes(im_arr, 0, 2)
im_arr = np.expand_dims(im_arr, axis=-1)  # adding channel dim
im_arr = np.expand_dims(im_arr, axis=0)  # adding batches dim

affine_matrix = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32)
affine_matrix = np.expand_dims(affine_matrix, axis=0)  # adding batches dim

ST = SpatialTransformer(interp_method='linear', indexing='ij')
x = ST([tf.convert_to_tensor(im_arr, np.float32),
        tf.convert_to_tensor(affine_matrix, np.float32)])

plt.imshow(im_arr[0, ..., 90, 0])
plt.colorbar()
plt.show()

plt.imshow(x[0, ..., 90, 0])
plt.colorbar()
plt.show()
