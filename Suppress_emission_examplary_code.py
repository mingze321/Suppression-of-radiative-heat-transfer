import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "dielectric_functions"

EPOCHS = 1500
NUM_TRIALS = 10
BATCH_SIZE = 256
SHUFFLE_BUFFER = 5000
INITIAL_LEARNING_RATE = 0.01
MAX_THICKNESS_NM = 1350.0
INCIDENT_ANGLE_DEG = 30.0


def configure_tensorflow():
    tf.keras.backend.set_floatx("float64")
    tf.debugging.set_log_device_placement(False)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs visible to TensorFlow: {gpus}")


def load_regular_material_grid(filename):
    data = np.genfromtxt(DATA_DIR / filename, skip_header=False)
    frequency = data[:, 0]
    real = data[:, 1]
    imag = data[:, 2]

    sort_order = np.argsort(frequency)
    frequency = frequency[sort_order]
    real = real[sort_order]
    imag = imag[sort_order]

    full_frequency = np.arange(400, 6000, 1, dtype=np.float64)
    full_real = np.interp(full_frequency, frequency, real)
    full_imag = np.interp(full_frequency, frequency, imag)

    return (
        tf.constant(full_real, dtype=tf.float64),
        tf.constant(full_imag, dtype=tf.float64),
    )


def interp_regular_1d(x, y_ref, x_ref_min=400.0, x_ref_step=1.0):
    x = tf.cast(x, tf.float64)
    y_ref = tf.cast(y_ref, tf.float64)
    max_index = tf.shape(y_ref)[0] - 2

    position = (x - x_ref_min) / x_ref_step
    lower_index = tf.cast(tf.floor(position), tf.int32)
    lower_index = tf.clip_by_value(lower_index, 0, max_index)
    upper_index = lower_index + 1

    lower_x = x_ref_min + tf.cast(lower_index, tf.float64) * x_ref_step
    weight = (x - lower_x) / x_ref_step
    weight = tf.clip_by_value(weight, 0.0, 1.0)

    lower_y = tf.gather(y_ref, lower_index)
    upper_y = tf.gather(y_ref, upper_index)
    return lower_y + weight * (upper_y - lower_y)


def material_index_from_grid(wavelength, real_grid, imag_grid):
    k = 10000000.0 / wavelength
    real = interp_regular_1d(k, real_grid)
    imag = interp_regular_1d(k, imag_grid)
    return tf.complex(real, imag)


def ch_matrix(theta_in, n0, n1, d1, n2, k0):
    one_m = tf.constant(1.0, tf.float64)
    zeros_m = tf.constant(0.0, tf.float64)
    imag = tf.complex(zeros_m, one_m)

    width, _ = n0.shape

    zeros_temp = tf.constant(0.0, dtype=tf.float64, shape=(width, 1))
    zeros_complex = tf.complex(zeros_temp, zeros_m)

    ones_temp = tf.constant(1.0, dtype=tf.float64, shape=(width, 1))
    ones_complex = tf.complex(ones_temp, zeros_m)

    k0 = tf.complex(k0, zeros_m)
    di = tf.complex(d1, zeros_m)
    di = tf.reshape(di, (width, 1))
    n1 = tf.reshape(n1, (width, 1))
    n2 = tf.reshape(n2, (width, 1))
    cos1 = 1 / n1 * tf.math.sqrt(n1 ** 2 - (n0 * tf.math.sin(theta_in)) ** 2)
    cos2 = 1 / n2 * tf.math.sqrt(n2 ** 2 - (n0 * tf.math.sin(theta_in)) ** 2)

    # S-polarized Fresnel coefficients. Use the commented equations below for P-polarization.
    rs12 = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
    ts12 = 2 * n1 * cos1 / (n1 * cos1 + n2 * cos2)
    # rs12 = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
    # ts12 = 2 * n1 * cos1 / (n2 * cos1 + n1 * cos2)

    ts12 = tf.expand_dims(ts12, axis=2)

    optical_pass = di * k0 * n1 * cos1
    matrix = [
        [tf.math.exp(-imag * optical_pass), zeros_complex],
        [zeros_complex, tf.math.exp(imag * optical_pass)],
    ]
    matrix = tf.reshape(matrix, shape=(4, width))
    matrix = tf.reshape(matrix, shape=(2, 2, width))
    matrix = tf.transpose(matrix, [2, 1, 0])

    interface_matrix = [[ones_complex, rs12], [rs12, ones_complex]]
    interface_matrix = tf.reshape(interface_matrix, shape=(4, width))
    interface_matrix = tf.reshape(interface_matrix, shape=(2, 2, width))
    interface_matrix = tf.transpose(interface_matrix, [2, 1, 0])

    return tf.linalg.matmul(matrix, interface_matrix) / ts12


def tmm(theta_in, n0, n, d, k0):
    width, layer_count = n.shape
    layer_count = layer_count - 1

    one_m = tf.constant(1.0, tf.float64, shape=[width, 1])
    zeros_m = tf.constant(0.0, tf.float64, shape=[width, 1])
    one_m_complex = tf.complex(one_m, zeros_m)

    cos0 = tf.math.cos(theta_in)
    cos1 = 1 / n[0, 0] * tf.math.sqrt(n[0, 0] ** 2 - (n0 * tf.math.sin(theta_in)) ** 2)

    # S-polarized Fresnel coefficients. Use the commented equations below for P-polarization.
    ts01 = 2 * n0 * cos0 / (n0 * cos0 + n[0, 0] * cos1)
    ts01 = tf.expand_dims(ts01, axis=2)
    rs01 = (n0 * cos0 - n[0, 0] * cos1) / (n0 * cos0 + n[0, 0] * cos1)
    # ts01 = 2 * n0 * cos0 / (n[0, 0] * cos0 + n0 * cos1)
    # ts01 = tf.expand_dims(ts01, axis=2)
    # rs01 = (n[0, 0] * cos0 - n0 * cos1) / (n[0, 0] * cos0 + n0 * cos1)

    interface_01 = [[one_m_complex, rs01], [rs01, one_m_complex]]
    interface_01 = tf.reshape(interface_01, shape=(4, width))
    interface_01 = tf.reshape(interface_01, shape=(2, 2, width))
    interface_01 = tf.transpose(interface_01, [2, 1, 0])

    transfer = interface_01 / ts01

    for layer_ind in range(layer_count):
        layer_transfer = ch_matrix(
            theta_in,
            n0,
            n[:, layer_ind],
            d[:, layer_ind],
            n[:, layer_ind + 1],
            k0,
        )
        transfer = tf.linalg.matmul(transfer, layer_transfer)

    return transfer


def reflectance(wavelength, theta_in, n0, n, d):
    zeros_m = tf.constant(0.0, tf.float64)

    theta_in = tf.complex(theta_in, zeros_m)
    k0 = 1 / wavelength * 2 * np.pi
    k0 = tf.expand_dims(k0, axis=1)

    matrix_overall = tmm(theta_in, n0, n, d, k0)

    m00 = matrix_overall[:, 0, 0]
    m10 = matrix_overall[:, 1, 0]
    refte = m10 / m00
    return tf.math.abs(refte) ** 2


def blackbody(wavelength):
    c = 3e8
    h = 6.625e-34
    k = 1.38e-23

    lam = wavelength * 1e-9
    v = c / lam

    temp_hot = 500.0
    hot = (8 * h * v ** 3) / ((c ** 3) * (tf.math.exp((h * v) / (k * temp_hot)) - 1)) * 1e19

    temp_cold = 300.0
    cold = (8 * h * v ** 3) / ((c ** 3) * (tf.math.exp((h * v) / (k * temp_cold)) - 1)) * 1e19

    return hot - cold


def set_target():
    wavelength_nm = np.arange(400, 3000, 2, dtype=np.float64)
    wavelength_nm = np.reshape(wavelength_nm, [-1, 1])
    target = np.zeros_like(wavelength_nm, dtype=np.float64)

    important_id = np.where(target < 80)
    wave_important = wavelength_nm[important_id]
    ref_important = target[important_id]

    return 10000000.0 / wavelength_nm, target, 10000000.0 / wave_important, ref_important


class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.layer_length = 29
        self.A = self.add_weight(
            name="A",
            shape=(self.layer_length,),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            trainable=True,
            dtype=tf.float64,
        )
        self.B = self.add_weight(
            name="B",
            shape=(self.layer_length,),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            trainable=True,
            dtype=tf.float64,
        )

        self.si_real_grid, self.si_imag_grid = load_regular_material_grid("Silicon_thin_film_NK_franta.txt")
        self.sio2_real_grid, self.sio2_imag_grid = load_regular_material_grid("SiO2_NK_franta.txt")

    def si_index(self, wavelength):
        return material_index_from_grid(wavelength, self.si_real_grid, self.si_imag_grid)

    def sio2_index(self, wavelength):
        return material_index_from_grid(wavelength, self.sio2_real_grid, self.sio2_imag_grid)

    def layer_thicknesses(self):
        d1 = tf.keras.activations.sigmoid(self.A[-self.layer_length:]) * MAX_THICKNESS_NM
        d2 = tf.keras.activations.sigmoid(self.B[-self.layer_length:]) * MAX_THICKNESS_NM
        return d1, d2

    def call(self, x):
        frequency_length = tf.size(x)
        x = tf.reshape(x, [frequency_length, 1])
        one_m = tf.constant(1.0, tf.float64)
        zeros_m = tf.constant(0.0, tf.float64)
        x = tf.cast(x, tf.float64)

        n0 = tf.complex(one_m, zeros_m)
        theta_in = one_m * INCIDENT_ANGLE_DEG / 180.0 * np.pi

        si = self.si_index(x)
        sio2 = self.sio2_index(x)
        substrate = sio2

        d1, d2 = self.layer_thicknesses()
        dielectric_stack = tf.concat([si, sio2, si, sio2, si, sio2, si], axis=1)
        _, physical_layer_count = dielectric_stack.shape

        d1 = tf.broadcast_to(d1, [frequency_length, self.layer_length])
        d2 = tf.broadcast_to(d2, [frequency_length, self.layer_length])
        n0 = tf.broadcast_to(n0, [frequency_length, 1])
        substrate = tf.broadcast_to(substrate, [frequency_length, 1])
        x = tf.reshape(x, [frequency_length])

        n_stack = tf.concat([dielectric_stack, substrate, substrate, substrate, substrate], axis=1)

        zero_thickness_padding = tf.constant([0.0, 0.0, 0.0], tf.float64)
        zero_thickness_padding = tf.broadcast_to(zero_thickness_padding, [frequency_length, 3])
        d3 = tf.concat([d1, zero_thickness_padding], axis=1)
        d4 = tf.concat([d2, zero_thickness_padding], axis=1)

        absorp_1 = 1 - reflectance(x, theta_in, n0, n_stack, d3)
        absorp_2 = 1 - reflectance(x, theta_in, n0, n_stack, d4)
        combined_absorp = tf.math.abs(
            absorp_1 * absorp_2 / (absorp_1 + absorp_2 - absorp_1 * absorp_2)
        ) * blackbody(x)

        self._export_t_1 = self.layer_thicknesses()[0][0:physical_layer_count]
        self._export_t_2 = self.layer_thicknesses()[1][0:physical_layer_count]

        return combined_absorp, blackbody(x), absorp_1, absorp_2

    def export_thicknesses(self):
        if not hasattr(self, "_export_t_1"):
            d1, d2 = self.layer_thicknesses()
            return d1[:7], d2[:7]
        return self._export_t_1, self._export_t_2


def linf(ypred, y):
    return tf.reduce_max((ypred - y) ** 2)


def train_one_trial(train_ds):
    model = MyModel()
    loss_object = tf.keras.losses.MeanSquaredError()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        INITIAL_LEARNING_RATE,
        decay_steps=500,
        decay_rate=0.7,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    train_loss = tf.keras.metrics.Mean(name="train_loss")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            prediction, _, _, _ = model(images, training=True)
            loss = loss_object(labels, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    for epoch in range(EPOCHS):
        train_loss.reset_state()
        for images, labels in train_ds:
            train_step(images, labels)

        if epoch % 600 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}, Loss: {train_loss.result()}")

    return model


def plot_results(wavelength, ref_target, abs1, abs2, abs3, abs4):
    plt.figure()
    plt.plot(10000000.0 / wavelength, ref_target, label="target")
    plt.plot(10000000.0 / wavelength, abs1, label="absorption from inverse design1")
    plt.plot(10000000.0 / wavelength, abs3, label="abs meta 1")
    plt.plot(10000000.0 / wavelength, abs4, label="abs meta 2")

    print("sum of the absorption", tf.reduce_sum(abs1).numpy())

    plt.title("designed")
    plt.xlabel("wavenumber(cm-1)")
    plt.ylabel("Absorption(%)")
    plt.legend()

    plt.figure()
    plt.plot(10000000.0 / wavelength, abs2, label="bb spectrum")
    plt.xlabel("wavenumber(cm-1)")
    plt.ylabel("Power")
    plt.legend()
    plt.show()


def save_designed_spectra(wavelength, designed_spectra):
    output = np.zeros((len(wavelength), 2))
    output[:, 0] = 10000000.0 / wavelength[:, 0]
    output[:, 1] = np.asarray(designed_spectra)
    np.savetxt(SCRIPT_DIR / "designed_spectra.csv", output, delimiter=",")


def main():
    configure_tensorflow()

    wavelength, ref_target, _, _ = set_target()
    train_ds = (
        tf.data.Dataset.from_tensor_slices((wavelength, ref_target))
        .shuffle(SHUFFLE_BUFFER)
        .batch(BATCH_SIZE)
    )

    best_loss = np.inf
    best_d1 = None
    best_d2 = None
    best_outputs = None

    for trial_id in range(NUM_TRIALS):
        model = train_one_trial(train_ds)
        abs1, abs2, abs3, abs4 = model(wavelength)

        current_loss = tf.reduce_sum(abs1).numpy()
        if current_loss < best_loss:
            best_loss = current_loss
            best_d1, best_d2 = model.export_thicknesses()
            best_outputs = (abs1, abs2, abs3, abs4)
            print("replaced", trial_id, best_d1.numpy(), best_d2.numpy(), current_loss)

    print("Final", NUM_TRIALS - 1, best_d1.numpy(), best_d2.numpy(), best_loss)

    abs1, abs2, abs3, abs4 = best_outputs
    plot_results(wavelength, ref_target, abs1, abs2, abs3, abs4)
    save_designed_spectra(wavelength, abs1)


if __name__ == "__main__":
    main()
