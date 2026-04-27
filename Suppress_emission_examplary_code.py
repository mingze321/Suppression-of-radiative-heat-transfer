import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
from tensorflow.keras import Model
import tensorflow_probability as tfp
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.interpolate import interp1d

# import pylab as pl

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(False)





# Define a single layer transfer matrix
def Ch_Matrix(theta_in, n0, n1, d1, n2, k0):
    one_m = tf.constant(1.0, tf.float64)
    zeros_m = tf.constant(0.0, tf.float64)
    imag = tf.complex(zeros_m, one_m)

    W, L = n0.shape

    zeros_temp = tf.constant(0.0, dtype=tf.float64, shape=(W, 1))
    zeros_complex = tf.complex(zeros_temp, zeros_m)

    ones_temp = tf.constant(1.0, dtype=tf.float64, shape=(W, 1))
    ones_complex = tf.complex(ones_temp, zeros_m)

    #     theta_in=tf.complex(theta_in,zeros_m)

    k0 = tf.complex(k0, zeros_m)
    di = tf.complex(d1, zeros_m)
    di = tf.reshape(di, (W, 1))
    n1 = tf.reshape(n1, (W, 1))
    n2 = tf.reshape(n2, (W, 1))
    cos1 = 1 / n1 * tf.math.sqrt(n1 ** 2 - (n0 * tf.math.sin(theta_in)) ** 2)
    cos2 = 1 / n2 * tf.math.sqrt(n2 ** 2 - (n0 * tf.math.sin(theta_in)) ** 2)
    # print (cos1, 'cos1')


    # IMPORTANT the commented code are for S-POL light
    rs12= (n1*cos1-n2*cos2)/ (n1*cos1+n2* cos2)
    ts12= 2*n1*cos1/ (n1*cos1+n2* cos2)
    # # IMPORTANT the following code are for P-POL light
    # rs12 = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
    # ts12 = 2 * n1 * cos1 / (n2 * cos1 + n1 * cos2)

    ts12 = tf.expand_dims(ts12, axis=2)

    optical_pass = di * k0 * n1 * cos1
    # print (optical_pass.shape, di.shape, k0.shape, n1.shape, cos1.shape, W, 'all shapes')
    Matrix = [[tf.math.exp(-imag * optical_pass), zeros_complex], [zeros_complex, tf.math.exp(imag * optical_pass)]]
    Matrix = tf.reshape(Matrix, shape=(4, W))
    Matrix = tf.reshape(Matrix, shape=(2, 2, W))
    Matrix = tf.transpose(Matrix, [2, 1, 0])

    M_t2 = [[ones_complex, rs12], [rs12, ones_complex]]
    M_t2 = tf.reshape(M_t2, shape=(4, W))
    M_t2 = tf.reshape(M_t2, shape=(2, 2, W))
    M_t2 = tf.transpose(M_t2, [2, 1, 0])

    M2 = tf.linalg.matmul(Matrix, M_t2) / ts12

    return M2


# Define the model of the TMM
# Define the model of the TMM
def TMM(theta_in, n0, n, d, k0):
    #     L=tf.size(n)
    #     print (n)
    temp = n
    W, L = temp.shape
    L = L - 1

    one_m = tf.constant(1.0, tf.float64, shape=[W, 1])
    zeros_m = tf.constant(0.0, tf.float64, shape=[W, 1])
    one_m_complex = tf.complex(one_m, zeros_m)

    cos0 = tf.math.cos(theta_in)
    cos1 = 1 / n[0, 0] * tf.math.sqrt(n[0, 0] ** 2 - (n0 * tf.math.sin(theta_in)) ** 2)


    # IMPORTANT the commented code are for S-POL light
    ts01=2*n0*cos0/ (n0*cos0 +n[0,0] *cos1)
    ts01=tf.expand_dims(ts01, axis=2)
    rs01= (n0*cos0 -n[0,0] *cos1)/(n0*cos0 +n[0,0] *cos1)
    # IMPORTANT the following module are for P-POL light
    # ts01=2*n0*cos0/ (n[0,0]*cos0 +n0 *cos1)
    # ts01=tf.expand_dims(ts01, axis=2)
    # rs01= (n[0,0]*cos0 -n0 *cos1) / (n[0,0]*cos0 +n0 *cos1)

    M01 = [[one_m_complex, rs01], [rs01, one_m_complex]]
    M01 = tf.reshape(M01, shape=(4, W))
    M01 = tf.reshape(M01, shape=(2, 2, W))
    M01 = tf.transpose(M01, [2, 1, 0])

    M0 = M01 / ts01
    Transfer = M0

    for layer_ind in range(L):
        Temp_transfer_matrix = Ch_Matrix(theta_in, n0, n[:, layer_ind], d[:, layer_ind], n[:, layer_ind + 1], k0)
        # print ('transfer matrix',Temp_transfer_matrix)
        # print ('medium transfer', layer_ind, Temp_transfer_matrix)
        Transfer = tf.linalg.matmul(Transfer, Temp_transfer_matrix)

    #         Transfer=Transfer * Temp_transfer_matrix
    return Transfer


# Reflectance calculation for Porous Si
# Reflectance calculation for Porous Si
# Reflectance calculation for Porous Si
def reflectance(wavelength, theta_in, n0, n, d1):
    one_m = tf.constant(1.0, tf.float64)
    zeros_m = tf.constant(0.0, tf.float64)

    #    All units here are in nm and nm-1
    d = d1

    # print (ns.shape, 'nsshape')

    theta_in = tf.complex(theta_in, zeros_m)
    # cons1 = tf.complex(one_m, zeros_m)

    k0 = 1 / wavelength * 2 * 3.1415926
    k0 = tf.expand_dims(k0, axis=1)

    # print ('n', n, 'thickness_nm', d1)

    Matrix_overall = TMM(theta_in, n0, n, d, k0)
    # print(Matrix_overall.shape, 'final shape')

    # print ('Final matrix', Matrix_overall)

    m00 = Matrix_overall[:, 0, 0]
    m10 = Matrix_overall[:, 1, 0]
    refte = m10 / m00
    t_te = 1 / m00

    # print ('ref', refte, 'wavelength', wavelength, m00, m10)

    #     Reflectance=tf.real(refte**2)
    Reflectance = (tf.math.abs(refte)) ** 2
    Transmission = (tf.math.abs(t_te)) ** 2 * tf.math.real(n[:, -1])

    #     Reflectance=Ref
    # print ('Y0',Y0, 'Ys', Ys, 'refte',refte)
    #     return Reflectance
    absorption1=(1-Reflectance- Transmission)*100
    return Reflectance




def Si_index(wavelength):
    k = 10000000 / wavelength  # k is in wavenumber

    Ge_data = np.genfromtxt("dielectric_functions/Silicon_thin_film_NK_franta.txt"
                            , skip_header=False)


    Full_frequency = np.arange(400, 6000, 1)
    Ge_frequency = Ge_data[:, 0]
    Ge_real = Ge_data[:, 1]
    Ge_imag = Ge_data[:, 2]

    f_real = interp1d(Ge_frequency, Ge_real)
    f_imag = interp1d(Ge_frequency, Ge_imag)
    Full_real_inter = f_real(Full_frequency)
    Full_imag_inter = f_imag(Full_frequency)
    # print(Full_real_inter, 'aaaa')

    Ge_real_inter = tfp.math.interp_regular_1d_grid(k, x_ref_min=400., x_ref_max=6000., y_ref=Full_real_inter)
    Ge_imag_inter = tfp.math.interp_regular_1d_grid(k, x_ref_min=400., x_ref_max=6000., y_ref=Full_imag_inter)
    # print(k, "K here")
    # Ge_real_inter=np.interp (k, Ge_frequency,Ge_real)
    # Ge_imag_inter = np.interp(k, Ge_frequency, Ge_imag)


    Ge_permitivity = tf.complex(Ge_real_inter, Ge_imag_inter)
    Ge_refractive_index = Ge_permitivity

    return Ge_refractive_index



def SiO2_index(wavelength):
    k = 10000000 / wavelength  # k is in wavenumber

    # Ge_data= np.genfromtxt("dielectric_functions/AlOx_epsilon_v2.txt"
    #                   , skip_header=False)
    Ge_data = np.genfromtxt("dielectric_functions/SiO2_NK_franta.txt"
                            , skip_header=False)


    Full_frequency = np.arange(400, 6000, 1)
    Ge_frequency = Ge_data[:, 0]
    Ge_real = Ge_data[:, 1]
    Ge_imag = Ge_data[:, 2]

    f_real = interp1d(Ge_frequency, Ge_real)
    f_imag = interp1d(Ge_frequency, Ge_imag)
    Full_real_inter = f_real(Full_frequency)
    Full_imag_inter = f_imag(Full_frequency)

    Ge_real_inter = tfp.math.interp_regular_1d_grid(k, x_ref_min=400., x_ref_max=6000., y_ref=Full_real_inter)
    Ge_imag_inter = tfp.math.interp_regular_1d_grid(k, x_ref_min=400., x_ref_max=6000., y_ref=Full_imag_inter)

    # Ge_real_inter=np.interp (k, Ge_frequency,Ge_real)
    # Ge_imag_inter = np.interp(k, Ge_frequency, Ge_imag)

    Ge_permitivity = tf.complex(Ge_real_inter, Ge_imag_inter)
    Ge_refractive_index = Ge_permitivity

    return Ge_refractive_index


def blackbody(wavelength):


    c = 3e+8

    h = 6.625e-34
    k = 1.38e-23
    T = 500.0
    Lam=wavelength*1e-9
    v=c/Lam

    I2=(8 * h * v**3) / ((c** 3) * (tf.math.exp((h * v)/ (k * T)) - 1))*1e+19

    T = 300.0
    I1 = (8 * h * v ** 3) / ((c ** 3) * (tf.math.exp((h * v) / (k * T)) - 1)) * 1e+19

    # print (wavelength)
    # k = 10000000 / wavelength  # k is in wavenumber


    return I2-I1





def set_target():
    # The following set target function is to set a gaussian shaped target

    # Define the frequency range and the target spectra
    # wavenumber_unshuffled =
    wavelength_unshuffled = np.arange(400,3000, 2)
    frequency_length = len(wavelength_unshuffled)
    wavelength_unshuffled = wavelength_unshuffled.reshape(frequency_length, )
    wavelength_unshuffled = np.float64(wavelength_unshuffled)
    wavelength_plot = wavelength_unshuffled.astype(np.float64)


    Ref_target_uns = np.ones((frequency_length, 1)) * 000



    Ref_target_uns = Ref_target_uns

    wavelength_plot = np.reshape(wavelength_plot, [frequency_length, 1])
    important_id = np.where(Ref_target_uns <80)
    wave_important = wavelength_plot[important_id]
    Ref_important = Ref_target_uns[important_id]

    # print(len(wave_important), 'data point in resonance')
    # plt.figure()
    # plt.plot( wavelength_plot, Ref_target_uns)
    # plt.xlabel('wavelength nm')
    # plt.ylabel('Reflectivity (%)')
    #
    # plt.title('target spectra')
    # plt.show()

    return 10000000/wavelength_plot, Ref_target_uns, 10000000/wave_important, Ref_important

class MyModel(Model):
    # class MyModel (Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer_length =29
        self.a = np.random.random(size=self.layer_length) - 0.5
        self.A = tf.Variable(self.a, trainable=True, dtype=tf.float64)

        self.b = np.random.random(size=self.layer_length) - 0.5
        self.B = tf.Variable(self.b, trainable=True, dtype=tf.float64)


        # self.Carrier=np.random.random(size=2)
        self.Carrier = np.random.random(size=4) - 0.5
        self.Carrier = tf.Variable(self.Carrier, trainable=True, dtype=tf.float64)
        self.Si_index=Si_index
        self.SiO2_index = SiO2_index
        self.blackbody=blackbody




    def call(self, x):
        frequency_length = tf.size(x)
        x = tf.reshape(x, [frequency_length, 1])
        one_m = tf.constant(1.0, tf.float64)
        zeros_m = tf.constant(0.0, tf.float64)
        x = tf.dtypes.cast(x, tf.float64)
        carrier1 = self.Carrier
        # carrier1 = tf.keras.activations.relu(carrier1,max_value=10) * 4 + 0.4
        carrier1 = tf.keras.activations.sigmoid(carrier1) * 3.6+0.4



        n0 = tf.complex(one_m, zeros_m)
        theta_in = one_m * 30 / 180 * 3.1415926

        Ge =  self.Si_index(x)
        SiO =self.SiO2_index(x)
        sub = SiO

        # print (Ge, 'Ge dielectric function')

        d = self.A[-self.layer_length:]
        d1 = tf.keras.activations.sigmoid(d) *1350.


        d_2 = self.B[-self.layer_length:]
        d2 = tf.keras.activations.sigmoid(d_2) *1350.


        # print(d1, 'thickness information')
        # print(d2, 'thickness information_t2')



        #Here is for real material property
        dielectric1 = tf.concat([Ge, SiO, Ge, SiO, Ge,SiO, Ge ], axis=1)
        dielectric1 = tf.concat([Ge, SiO, Ge, SiO, Ge, SiO, Ge], axis=1)
        # Here is for disperseless material property
        # dielectric1 = [Ge, SiO, Ge, SiO, Ge]
        # dielectric1 = tf.broadcast_to(dielectric1, [frequency_length, self.layer_length])
        W, L = dielectric1.shape


        global export_t_1,export_t_2

        export_t_1=d1[0:L]
        export_t_2 = d2[0:L]

        d1 = tf.broadcast_to(d1, [frequency_length, self.layer_length])
        d2 = tf.broadcast_to(d2, [frequency_length, self.layer_length])
        n0 = tf.broadcast_to(n0, [frequency_length, 1])
        ns = tf.broadcast_to(sub, [frequency_length, 1])
        ns_lossless=tf.broadcast_to(sub, [frequency_length, 1])
        x = tf.reshape(x, [frequency_length, ])

        n2 = tf.concat([dielectric1, ns, ns, ns, ns], axis=1)

        thickness_cdo = tf.constant([0,0,0], tf.float64)
        d_cdo = tf.broadcast_to(thickness_cdo, [frequency_length, 3])
        d3 = tf.concat([d1, d_cdo], axis=1)
        d4 = tf.concat([d2, d_cdo], axis=1)
        # print(thickness_cdo, 'thickness of cdo')

        absorp_1 = (1-reflectance(x, theta_in, n0, n2, d3))
        absorp_2 = (1 - reflectance(x, theta_in, n0, n2, d4))

        absorp1=tf.math.abs(absorp_1*absorp_2/(absorp_1+absorp_2-absorp_1*absorp_2))*self.blackbody(x)





        return absorp1,   self.blackbody(x),absorp_1,absorp_2


def Linf(ypred, y):
    res = (ypred - y) ** 2
    res = tf.reduce_max(res)
    return res


# set the target spectra
wavelength, ref_target, wavelength_important, ref_important = set_target()


train_ds = tf.data.Dataset.from_tensor_slices((wavelength, ref_target)).shuffle(5000).batch(256)



EPOCHS = 1500

number_of_trails=10
best_loss=100000
best_d1=[]
best_d2=[]
for trails_id in range (number_of_trails):

    # Define the model
    model = MyModel()
    tf.keras.backend.set_floatx('float64')
    loss_object = tf.keras.losses.MeanSquaredError()
    loss_object1 = tf.keras.losses.MeanAbsoluteError()
    loss_object3 = Linf

    initial_learning_rate = 0.01
    lr_schedule2 = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=500,
        decay_rate=0.7,
        staircase=True)


    optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr_schedule2)
    # optimizer=tf.keras.optimizers.SGD (learning_rate=10.0)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')



    @tf.function
    def train_step1(images, labels1):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            print(model.trainable_variables, 'all variables')
            prediction1, prediction2, prediction3, prediction4 = model(images, training=True)

            loss = loss_object(labels1, prediction1)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer2.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)






    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        for images, labels1 in train_ds:
            train_step1(images, labels1)

        if epoch % 600 == 0 or epoch == 0:
            template = 'Epoch {}, Loss: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  ))

    frequency_len = len(wavelength)

    wavelength_wide =   np.arange(100,20000, 10)
    wavelength_wide=10000000/wavelength_wide
    wavelength_wide = np.reshape(wavelength_wide, [1990, 1])

    abs1,abs2,abs3,abs4 = model(wavelength)
    designed_spectra =abs1

    current_loss=sum(abs1)
    if current_loss<best_loss:
        best_loss=current_loss
        best_d1=export_t_1
        best_d2=export_t_2
        print ("replaced",trails_id, best_d1,best_d2,current_loss)


print ("Final",trails_id, best_d1,best_d2,best_loss)















# ref_target_plot=(ref_target-ref_target.min())/(ref_target.max()-ref_target.min())
plt.figure()
# plt.plot(10000000/wavelength, designed_spectra, label='Spectra from inverse design', linewidth=6)
# plt.plot (10000000/wavelength, ref_target, label='target spectra', linewidth=0.5)

plt.plot(10000000/wavelength,  ref_target, label='target')
# plt.plot(10000000 / wavelength, transmission, label='transmission from inverse design')
plt.plot( 10000000/wavelength, abs1, label='absoprtion from inverse design1')
plt.plot( 10000000/wavelength, abs3, label='abs meta 1')
plt.plot( 10000000/wavelength, abs4, label='abs meta 2')


print("sum of the absorption",sum(abs1))

plt.title('designed')
plt.xlabel('wavenumber(cm-1)')
plt.ylabel('Absorption(%)')
plt.legend()


plt.figure()


plt.plot( 10000000/wavelength, abs2, label='bb spectrum')

plt.xlabel('wavenumber(cm-1)')
plt.ylabel('Power')
plt.legend()





plt.show()






Designed_output = np.zeros((frequency_len, 2))
Designed_output[:, 0] = 10000000/wavelength[:, 0]
Designed_output[:, 1] = designed_spectra
# Designed_output[:, 2] = transmission
#
# print(model.trainable_variables)
# ind = 0
# # temp=np.zeros((12,12))
#
#
# # np.savetxt('tam_designed.csv',designed_structure, delimiter=",")

np.savetxt('designed_spectra.csv', Designed_output, delimiter=",")
# # np.savetxt('tam_designed.csv',designed_structure, delimiter=",")
