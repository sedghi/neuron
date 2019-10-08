import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from neuron.layers import SpatialTransformer
from pathlib import Path
import tensorflow.keras.backend as K

from neuron.metrics import MeanSquaredError


def transform_image(image, transform, reference_image, default_value=None, interpolator=sitk.sitkLinear):
    """
    transform an image with a given sitk transform
    :param image: sitk image
    :param transform: sitk transfrom to be used
    :param reference_image: reference image used for resampling
    :param default_value: default value pixel value if needed
    :param interpolator: interpolator
    :return: transformed image
    """
    if default_value == None:
        default_value = reference_image[0, 0, 0]
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def rescale_zero_one(image):
    """
    rescale intensities of an image to [0,1]
    :param image: sitk image
    :return: rescaled image to [0,1]
    """
    return sitk.RescaleIntensity(sitk.Cast(image, sitk.sitkFloat32), 0, 1)


def read_sample_images(random_tx=False, rescale=False):
    """
    Read a sample images for the experiments
    :param random_tx: random transform applied to the moving image if desired
    :param rescale: rescaled intensities if desired
    :return: image array with size (1,w,h,d,1), first dim is for batch size, last dim
    for channels in tensorflow.
    """
    # fixed
    im_fixed = sitk.ReadImage(str(Path.cwd().joinpath("case051_T1.nii.gz")))
    if rescale_zero_one:
        im_fixed = rescale_zero_one(im_fixed)
    im_arr_fixed = sitk.GetArrayFromImage(im_fixed)
    im_arr_fixed = np.swapaxes(im_arr_fixed, 0, 2)  # need to swap axes because of sitk and numpy array order mismatch
    im_arr_fixed = np.expand_dims(im_arr_fixed, axis=-1)
    im_arr_fixed = np.expand_dims(im_arr_fixed, axis=0)

    if random_tx:
        # random translation transform applied to the image
        tx = sitk.AffineTransform(3)
        tx.SetTranslation([10, 0, 0])
        im_moving = transform_image(im_fixed, tx, im_fixed)
    else:
        im_moving = im_fixed

    im_arr_moving = sitk.GetArrayFromImage(im_moving)
    im_arr_moving = np.swapaxes(im_arr_moving, 0, 2)
    im_arr_moving = np.expand_dims(im_arr_moving, axis=-1)
    im_arr_moving = np.expand_dims(im_arr_moving, axis=0)
    return im_arr_fixed, im_arr_moving, im_fixed.GetSpacing()


def create_identity_transform_stn():
    """
    create a sample Identity transformation matrix compatible with STN.
    :return: 3D affine matrix in shape of (1,3,4) , 1 is for batch size
    """
    affine_matrix = np.array([1, 0, 0, 0,
                              0, 1, 0, 0,
                              0, 0, 1, 0], dtype=np.float32)
    affine_matrix_diff_from_identity = affine_matrix - ([1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    affine_matrix_diff_from_identity = affine_matrix_diff_from_identity.astype(np.float32)
    return np.reshape(affine_matrix_diff_from_identity, (1, 3, 4))


def plot_side2side(before, after, title):
    """
    plotting tool for before and after registration
    :param before: before transformation 3D array
    :param after: after transformation 3D array
    :param title: title of the plot
    :return:
    """
    n_slices = before.shape[-1]
    f, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    ax1, ax2 = axs.ravel()
    im1 = ax1.imshow(before[0, ..., n_slices // 2, 0])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.2)
    cbar1 = plt.colorbar(im1, cax=cax1, format="%.1f")
    im2 = ax2.imshow(after[0, ..., n_slices // 2, 0])
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.2)
    cbar2 = plt.colorbar(im2, cax=cax2, format="%.1f")
    f.suptitle(str(title), fontsize=17)
    plt.show()


def perform_test_affine_transform(arr):
    """
    perform an identity affine transformation for testing
    :param arr: image array
    :return:
    """
    tx = create_identity_transform_stn()
    ST = SpatialTransformer(interp_method='linear', indexing='ij')
    x = ST([tf.convert_to_tensor(arr), tf.convert_to_tensor(tx)])
    plot_side2side(arr, x, 'before after')
    return 1


class Registration():
    """
    registration class used for transformation optimization
    """

    def __init__(self, fixed, moving, x, y, z, masked_loss=True):
        """
        parameters of the class are
        :param fixed: fixed image array
        :param moving: moving image array to be registered
        :param x: translation in X as a trainable tf variable
        :param y: translation in Y as a trainable tf variable
        :param z: translation in Z as a trainable tf variable
        :param masked_loss: whether to calculate the loss wrt to the masked head only (remove bg)
        """
        self.stn = SpatialTransformer(interp_method='linear', indexing='ij')
        self.iteartion = 0
        self.fixed = fixed
        self.moving = moving
        self.masked_loss = masked_loss
        self.x = x
        self.y = y
        self.z = z

    def compute_loss(self):
        self.iteartion += 1
        rot_matrix = tf.Variable(initial_value=tf.eye(3), trainable=False)
        t = tf.reshape([self.x, self.y, self.z], (3, 1))
        affine_matrix = tf.concat([rot_matrix, t], axis=1)
        affine_matrix = tf.reshape(affine_matrix, (1, 3, 4))
        affine_matrix = affine_matrix - tf.Variable(initial_value=[[1, 0, 0, 0],
                                                                   [0, 1, 0, 0],
                                                                   [0, 0, 1, 0]], dtype=tf.float32, trainable=False)
        transformed_moving = self.stn([self.moving, affine_matrix])

        # plt.imshow(np.array(tf.cast(self.fixed > 0.05, tf.float32))[0, ..., 40, 0])
        # plt.title("fixed")
        # plt.colorbar()
        # plt.show()

        if self.masked_loss:
            mask_fixed = tf.cast(self.fixed > 0.05, tf.float32)  # good threshold for masking
            return K.mean(
                K.square(tf.boolean_mask(self.fixed, mask_fixed) - tf.boolean_mask(transformed_moving, mask_fixed)))
        else:
            return K.mean(K.square(self.fixed - transformed_moving))


if __name__ == "__main__":
    im_arr_fixed, im_arr_moving, spacings = read_sample_images(random_tx=True, rescale=True)
    # perform_test_affine_transform(arr=im_arr_moving)

    # performing gradient descent on the translation parameters in the following
    # we define 3 TF trainable variables, the loss function is MSE of the intensities in the images
    # because we are working on a uni-modal case it is a good Loss function to perform registration
    # moving image is shifted 10 mm in X direction, so a successful optimization needs to derive X variable
    # as -5 (since each voxel is 2mm in the image), we hope to start X variable from initial value of -1 and it
    # goes to -5
    optimizer = keras.optimizers.SGD(learning_rate=10)

    x = tf.Variable(-1, name='x', trainable=True, dtype=tf.float32)
    y = tf.Variable(0.0, name='y', trainable=True, dtype=tf.float32)
    z = tf.Variable(0, name='z', trainable=True, dtype=tf.float32)
    trainable_vars = [x, y, z]

    # creating registration model
    reg_model = Registration(fixed=im_arr_fixed, moving=im_arr_moving,
                             x=x, y=y, z=z, masked_loss=True)

    for i in range(1000):
        # minimizing the loss wrt the trainable variables that are x,y,z
        optimizer.minimize(reg_model.compute_loss, trainable_vars)
        print("x:{},y:{},z:{}".format(reg_model.x.numpy(),
                                      reg_model.y.numpy(),
                                      reg_model.z.numpy()))
