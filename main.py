import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from neuron.layers import SpatialTransformer
from pathlib import Path


def transform_image(image, transform, reference_image, default_value=None, interpolator=sitk.sitkLinear):
    if default_value == None:
        default_value = reference_image[0, 0, 0]
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def rescale_zero_one(image):
    image = sitk.RescaleIntensity(sitk.Cast(image, sitk.sitkFloat32), 0, 1)
    return image


def read_sample_images(random_tx=False, rescale=False):
    """
    Read a sample images (fixed and moving) for transformation
    :return: image array with size (1,w,h,d,1), first dim is for batch size,
    last dim is for channel (default tensorflow)
    """
    # fixed
    im_fixed = sitk.ReadImage(str(Path.cwd().joinpath("case051_T1.nii.gz")))
    if rescale_zero_one:
        im_fixed = rescale_zero_one(im_fixed)
    im_arr_fixed = sitk.GetArrayFromImage(im_fixed)
    im_arr_fixed = np.swapaxes(im_arr_fixed, 0, 2)
    im_arr_fixed = np.expand_dims(im_arr_fixed, axis=-1)
    im_arr_fixed = np.expand_dims(im_arr_fixed, axis=0)

    if random_tx:
        tx = sitk.AffineTransform(3)
        tx.SetTranslation([10, 0, 0])
        im_moving = transform_image(im_fixed, tx, im_fixed)
    else:
        im_moving = im_fixed

    im_arr_moving = sitk.GetArrayFromImage(im_moving)
    im_arr_moving = np.swapaxes(im_arr_moving, 0, 2)
    im_arr_moving = np.expand_dims(im_arr_moving, axis=-1)
    im_arr_moving = np.expand_dims(im_arr_moving, axis=0)
    return im_arr_fixed, im_arr_moving


def create_sample_translation_matrix():
    affine_matrix = np.array([1, 0, 0, 0,
                              0, 1, 0, 10,
                              0, 0, 1, 10], dtype=np.float32)
    affine_matrix_diff_from_identity = affine_matrix - ([1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    affine_matrix_diff_from_identity = affine_matrix_diff_from_identity.astype(np.float32)
    return np.reshape(affine_matrix_diff_from_identity, (1, 3, 4))


def plot_side2side(before, after, title):
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


def load_model():
    model_dir = Path.cwd().joinpath("model_dir")

    cnn_model = tf.keras.models.load_model(str(model_dir.joinpath('model_checkpoint.hdf5')), compile=False)
    for i in range(len(cnn_model.layers)):
        cnn_model.layers[i].trainable = False

    mean_npy = np.load(str(model_dir.joinpath("training_data_mean.npy")))
    std_npy = np.load(str(model_dir.joinpath("training_data_std.npy")))
    return cnn_model, mean_npy, std_npy


def perform_test_affine_transform(arr):
    tx = create_sample_translation_matrix()
    ST = SpatialTransformer(interp_method='linear', indexing='ij')
    x = ST([tf.convert_to_tensor(arr), tf.convert_to_tensor(tx)])
    plot_side2side(arr, x, 'before after')
    return 1


class Registration():
    def __init__(self, fixed, moving, x, y, z):
        self.stn = SpatialTransformer(interp_method='linear', indexing='ij')
        self.iteartion = 0
        self.fixed = fixed
        self.moving = moving
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

        plt.imshow(np.array(tf.square(self.fixed - transformed_moving))[0, ..., 40, 0])
        # plt.title(str(1))
        plt.colorbar()
        plt.show()

        return tf.reduce_sum(tf.square(self.fixed - transformed_moving)) / tf.cast(tf.size(self.fixed), tf.float32)


if __name__ == "__main__":
    mode = "train"
    im_arr_fixed, im_arr_moving = read_sample_images(random_tx=True, rescale=True)
    if mode == "test":
        perform_test_affine_transform(arr=im_arr_moving)
    else:
        # load cnn model and mean/std npys
        # cnn_model, mean_npy, std_npy = load_model()
        optimizer = keras.optimizers.SGD(learning_rate=1)
        # affine_matrix, trainable_vars = create_trainable_translation_tf_params()

        x = tf.Variable(0.01, name='x', trainable=True, dtype=tf.float32)
        y = tf.Variable(0.01, name='y', trainable=True, dtype=tf.float32)
        z = tf.Variable(0.01, name='z', trainable=True, dtype=tf.float32)
        trainable_vars = [x, y, z]

        reg_model = Registration(fixed=im_arr_fixed, moving=im_arr_moving,
                                 x=x, y=y, z=z)

        for i in range(1000):
            optimizer.minimize(reg_model.compute_loss, trainable_vars)
            print("x:{},y:{},z:{}".format(reg_model.x.numpy(),
                                          reg_model.y.numpy(),
                                          reg_model.z.numpy()))
            # with tf.GradientTape(persistent=True) as tape:
            #     loss = reg_model.compute_loss(im_arr_fixed, im_arr_moving, trainable_vars)
            # gradients = tape.gradient(loss, trainable_vars)
            # optimizer.apply_gradients(zip(gradients, trainable_vars))

        # plot_side2side(im_arr_moving, x, 'before after')
