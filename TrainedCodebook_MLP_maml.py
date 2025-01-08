import os
import argparse
import numpy as np
import numpy.matlib
import torch.nn as nn
import scipy.io as sio
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt

from meta import Meta                      # MAML Meta-learning scheme deployment
from DeepMIMONshot import deepMIMONshot    # load N-ways K-shots deepMIMO MAML data
from sklearn.model_selection import train_test_split            # to split dataset into train-test indexes
from beam_utils import ULA_DFT_codebook as DFT_codebook         # DFT codebook generator
from beam_utils import plot_codebook_pattern as plot_codebook   # for visulisation of codebook beamparttern


## (1) BS signal transmit power and noise power. (2) Torch device setting.

np.random.seed(233)  

tx_power_dBm = 20       
noise_factor = 0   # dB
noise_spectrum_density_dBm = -174   # dBm/Hz
BandWidth = 100e6    # 100 MHz
noise_power_dBm = noise_spectrum_density_dBm + 10*np.log10(BandWidth)
noiseless = False

if noiseless:
    noise_power_dBm = -np.inf    
noise_power = 10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

## Parameters setting
argparser = argparse.ArgumentParser(description = 'MetaBeam_train')
argparser.add_argument('--epoch', type  =int, help='epoch number', default = 2010)
argparser.add_argument('--n_way', type = int, help='N-ways in MAML', default = 1)                           # 5 by default
argparser.add_argument('--k_spt', type = int, help='K-shots for MAML support set', default = 5)             # 1 by default
argparser.add_argument('--k_qry', type = int, help='K-shots for MAML for query set', default = 20)          # 15 by default
argparser.add_argument('--task_num', type = int, help='Meta training batch size', default = 10)             # 32 by default
argparser.add_argument('--meta_lr', type = float, help='Outer update learning rate, using Adam', default = 1e-2)     # 1e-3 by default
argparser.add_argument('--update_lr', type = float, help='Inner update learning rate', default = 0.05)               # 1e-2 by default
argparser.add_argument('--update_step_train', type = int, help='Meta training inner update steps', default = 5)      # 5 by default
argparser.add_argument('--update_step_test', type = int, help='Finetuning process update steps', default = 10)        # 10 by default
argparser.add_argument('--n_antenna', type = int, help='Basestation ULA antennas number', default = 64)
argparser.add_argument('--n_wb', type = int, help='Trained codebook size (number of probing beams)', default = 12)
argparser.add_argument('--n_nb', type = int, help='Narrow DFT-beam number', default = 128)
argparser.add_argument('--tx_power_dBm', type = int, help='Tx power in dBm', default = tx_power_dBm)
argparser.add_argument('--noise_factor', type = int, help='Noise factor in dBm', default = noise_factor)
argparser.add_argument('--noise_power_dBm', type = int, help='Power of AWGN noise in dBm', default = noise_power_dBm)
argparser.add_argument('--noise_power', type = int, help='ower of AWGN noise', default = noise_power)
argparser.add_argument('--ifdyn', type = int, help='Whether use O2_DYN', default = False)

# args = argparser.parse_args()
args, unknow = argparser.parse_known_args()
print(args)

## * Load deepMIMO data from '.mat' or '.npy' documents in real and imag parts seperately.  
# * Dataset name structure:  
# e.g. 'O1_28B_BS3_1TO2751' means:  
# Outdoor scenario 1 with Block, under 28GHz frequency, activated row from 1 to 2751.

h_real_c = []
h_imag_c = []

# 'Ordering' dataset here. Concatenation will be done sequentially.'
dataset = ['O1_28_BS3_800TO1200', 'I3_60_BS1_1TO551']

for index, item in enumerate(dataset):
    dataset_name = item
    if args.ifdyn:
        root = './ZY_dataset/O2_DYN_3P5_1TO1891/{}/'.format(dataset_name)
        if not os.path.isfile(os.path.join(root, '{}_real.npy'.format(dataset_name))):
            fname_h_real = './ZY_dataset/O2_DYN_3P5_1TO1891/{}/h_real.mat'.format(dataset_name)
            fname_h_imag = './ZY_dataset/O2_DYN_3P5_1TO1891/{}/h_imag.mat'.format(dataset_name)
            print("Loading from {}.mat ...".format(dataset_name)) 
            h_real_temp = sio.loadmat(fname_h_real)['h_real'] # in shape of (100, n_row, n_antenna)  
            h_imag_temp = sio.loadmat(fname_h_imag)['h_imag'] # with 100 user grids per row
            h_real_temp = h_real_temp.transpose(1, 0, 2)      # transpose it -> (n_row, 100, n_antenna)
            h_imag_temp = h_imag_temp.transpose(1, 0, 2)      # 
            root = './ZY_dataset/O2_DYN_3P5_1TO1891/{}/'.format(dataset_name)
            print('Writing into {}.npy ...'.format(dataset_name))
            np.save(os.path.join(root, '{}_real.npy'.format(dataset_name)), h_real_temp)
            np.save(os.path.join(root, '{}_imag.npy'.format(dataset_name)), h_imag_temp)
            print('New sub-dataset shape:', h_real_temp.shape)
            
        else:
            print("Loading from {}.npy ...".format(dataset_name)) 
            h_real_temp = np.load(os.path.join(root, '{}_real.npy'.format(dataset_name)))
            h_imag_temp = np.load(os.path.join(root, '{}_imag.npy'.format(dataset_name)))
            print('Sub-dataset shape:', h_real_temp.shape) 
            
        if index == 0:  # if no sub-dataset been concatenated yet
            h_real_c = h_real_temp      # h_real_concatenation
            h_imag_c = h_imag_temp
        else:
            h_real_c = np.concatenate((h_real_c, h_real_temp), axis = 0)
            h_imag_c = np.concatenate((h_imag_c, h_imag_temp), axis = 0)
    else:
        root = './ZY_dataset/{}/'.format(dataset_name)
        if not os.path.isfile(os.path.join(root, '{}_real.npy'.format(dataset_name))):
            fname_h_real = './ZY_dataset/{}/h_real.mat'.format(dataset_name)
            fname_h_imag = './ZY_dataset/{}/h_imag.mat'.format(dataset_name)
            print("Loading from {}.mat ...".format(dataset_name)) 
            h_real_temp = sio.loadmat(fname_h_real)['h_real'] # in shape of (100, n_row, n_antenna)  
            h_imag_temp = sio.loadmat(fname_h_imag)['h_imag'] # with 100 user grids per row
            h_real_temp = h_real_temp.transpose(1, 0, 2)      # transpose it -> (n_row, 100, n_antenna)
            h_imag_temp = h_imag_temp.transpose(1, 0, 2)      # 
            root = './ZY_dataset/{}/'.format(dataset_name)
            print('Writing into {}.npy ...'.format(dataset_name))
            np.save(os.path.join(root, '{}_real.npy'.format(dataset_name)), h_real_temp)
            np.save(os.path.join(root, '{}_imag.npy'.format(dataset_name)), h_imag_temp)
            print('New sub-dataset shape:', h_real_temp.shape)
            
        else:
            print("Loading from {}.npy ...".format(dataset_name)) 
            h_real_temp = np.load(os.path.join(root, '{}_real.npy'.format(dataset_name)))
            h_imag_temp = np.load(os.path.join(root, '{}_imag.npy'.format(dataset_name)))
            print('Sub-dataset shape:', h_real_temp.shape) 
            
        if index == 0:  # if no sub-dataset been concatenated yet
            h_real_c = h_real_temp      # h_real_concatenation
            h_imag_c = h_imag_temp
        else:
            h_real_c = np.concatenate((h_real_c, h_real_temp), axis = 0)
            h_imag_c = np.concatenate((h_imag_c, h_imag_temp), axis = 0)
        
print('Ultimate dataset shape:', h_real_c.shape) 

h_real = h_real_c
h_imag = h_imag_c
h_cplx = h_real + 1j*h_imag
num_row, num_ue, _ = h_cplx.shape
print('Number of rows: {}, number of user grids per row: {}.'.format(num_row, num_ue))

## Label: dft codebook index that result in max SNR.  
# DFT codebook uniformly seperates the angular space into n_nb (e.g. 128) parts.

n_wb = args.n_wb
n_nb = args.n_nb
print('{} Wide Beams, {} Narrow Beams.'.format(n_wb, n_nb))

dft_nb_codebook = DFT_codebook(nseg = n_nb, n_antenna = args.n_antenna)
# fig, ax = plot_codebook(dft_nb_codebook)
plot_narrow_beam = dft_nb_codebook[[10, 60, 110], :]   
fig1, ax1 = plot_codebook(plot_narrow_beam)     # plot several DFT narrow beams        
label = np.argmax(np.power(np.absolute(np.matmul(h_cplx, dft_nb_codebook.conj().T)), 2), axis = 2)
soft_label = np.power(np.absolute(np.matmul(h_cplx, dft_nb_codebook.conj().T)), 2)

print(label.shape)

## Normalize on each user grid after adding noise. (normalization factor is the absolute value of the strongest received signal among 64 antennas)

# plot antennas received signal without AWGN
plt.subplot(211)
plt.plot(h_real[30, 50, :])
plt.plot(h_imag[30, 50, :])

noise_vec_real = np.random.normal(0, 1, size = h_real.shape)*np.sqrt(noise_power/2)
noise_vec_imag = np.random.normal(0, 1, size = h_imag.shape)*np.sqrt(noise_power/2)
h_real = h_real + noise_vec_real
h_imag = h_imag + noise_vec_imag
# plot antennas received signal with AWGN
plt.subplot(212)
plt.plot(h_real[30, 50, :])
plt.plot(h_imag[30, 50, :])
plt.show()

h_temp = (h_real + 1j*h_imag).reshape(-1, 64)
# Norm: among the 64 antennas, the absolute value of the strongest received signal
norm = np.max(abs(h_temp), axis = 1)    
norm_mtx = numpy.matlib.repmat(norm, 64, 1).transpose(1, 0)
h_temp = np.divide(h_temp, norm_mtx)            # normalization (maxima -> 1)
h_temp = h_temp.reshape(num_row, num_ue, -1)    # reshape h_temp into (n_row, n_ue, n_antenna)
print('h_temp shape: {}'.format(h_temp.shape))

## * Since pytorch does not support complex operations, so we concatenate the real and image part of received signal, forming into a (n_row, n_ue, 2*n_antenna) data matrix -- h_concat_scaled. 

h_concat_scaled = np.concatenate((np.real(h_temp), np.imag(h_temp)), axis = 2)
train_idx, test_idx = train_test_split(np.arange(num_row), test_size = 0.5, shuffle = False)

x_train, y_train = h_concat_scaled[train_idx, :], label[train_idx]
x_test, y_test = h_concat_scaled[test_idx, :], label[test_idx]
print("Shape of training data and label:", x_train.shape, y_train.shape)
print("Shape of test data and label:", x_test.shape, y_test.shape)

config = [  
            ('complexnn', [n_wb, args.n_antenna]), 
            ('linear', [2*n_wb, n_wb]),
            ('relu', [True]),
            ('linear', [3*n_wb, 2*n_wb]),
            ('relu', [True]),
            ('linear', [n_nb, 3*n_wb])
]

## Instantiate MAML as net and calculate total variable number that involved in gradient descent.

net = Meta(args, config, norm_factor = None).to(device)
print(net)

temp = filter(lambda x: x.requires_grad, net.parameters())  # Variables involved in gradient calculation
num = sum(map(lambda x: np.prod(x.shape), temp))            # np.prod returns the product of array elements over a given axis
print('Total trainable tensors:', num)

db_train = deepMIMONshot(args, x_train, y_train, x_test, y_test)

## * Dataloader which load MAML-MIMO (task_num, n_ways, )training data:  
# num of storage samples in cache * (n_ways, task_num, k_spt + k_qry, 2*n_antenna)  
# * Note tha different from Omniglot，we need to seperately consider data label. 

# plot the training and test data sample

x_train_sample = db_train.x_train[30, 10, :]
plt.subplot(221), plt.plot(x_train_sample)
y_train_sample = db_train.y_train[30, :]
plt.subplot(222), plt.plot(y_train_sample)
x_test_sample = db_train.x_test[10, 10, :]
plt.subplot(223), plt.plot(x_test_sample)
y_test_sample = db_train.y_test[10, :]
plt.subplot(224), plt.plot(y_test_sample), plt.show()

## Deepcopy in Fintuning does not function well. Suspect that the parameterization in ComplexNN has some problem. So directly save and load '.pkl' instead. 

trained_codebook = []
if_updatePlot = False

def plot_temp_beam(beam_weights):
    real_init = (1 / 8) * np.cos(beam_weights)  #
    imag_init = (1 / 8) * np.sin(beam_weights)  #        
    beam_weights_init = real_init + 1j*imag_init
    fig, ax = plot_codebook(beam_weights_init)

## Training process

for step in range(args.epoch):
    accs_train = []
    x_spt, y_spt, x_qry, y_qry = db_train.next('train')
    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                 torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
    train_accs = net(x_spt, y_spt, x_qry, y_qry)
    accs_train.append( train_accs )

    if step % 10 == 0:
        print('step:', step, '\ttraining acc:', train_accs)

    if step % 50 == 0:
        # append trained codebook
        # trained_codebook.append(net.net.get_codebook())
        temp_codebook = net.net.get_codebook()
        temp_codebook = np.array(temp_codebook)
        trained_codebook.append(temp_codebook)
        # fig, ax = plot_codebook(temp_codebook.transpose(1, 0))
        if if_updatePlot:
            fig, ax = plot_codebook(temp_codebook)

        accs_test = []
        thetas_init = []
        ws_init = []
        bs_init = []
        thetas_finetuned = []
        ws_finetuned = []
        bs_finetuned = []
        beam_weights_init = []
        beam_weights_finetuned = []
        # So that in average we can iterate over the whole test set, 
        # to better test the overall adaptation ability
        for _ in range(x_test.shape[0] // args.task_num):        
            # test
            x_spt, y_spt, x_qry, y_qry = db_train.next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                         torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc, theta_init, w_init, b_init, theta_finetuned, w_finetuned, b_finetuned = net.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                
                accs_test.append( test_acc )
                thetas_init.append(theta_init)
                ws_init.append(w_init)
                bs_init.append(b_init)
                thetas_finetuned.append(theta_finetuned)
                ws_finetuned.append(w_finetuned)
                bs_finetuned.append(b_finetuned)

        # [b, update_step+1]
        # !!!!!!!!!!!!!!!! accs itself is only with one sample !!!!!!!!!!!!!!!!
        accs_test = np.array(accs_test).mean(axis = 0).astype(np.float16)

        real_init = (1 / 8) * np.cos(thetas_init)  #
        imag_init = (1 / 8) * np.sin(thetas_init)  #        
        beam_weights_init = real_init + 1j*imag_init
        beam_weights_init = np.array(beam_weights_init).mean(axis = 0)

        real_finetuned = (1 / 8) * np.cos(thetas_finetuned)  #
        imag_finetuned = (1 / 8) * np.sin(thetas_finetuned)  #        
        beam_weights_finetuned = real_finetuned + 1j*imag_finetuned
        beam_weights_finetuned = np.array(beam_weights_finetuned).mean(axis = 0)
        # accs_train = np.array(accs_train).mean(axis = 0).astype(np.float16)
        print('step:', step, '\ttest acc:', accs_test)

## Visualization of the performance of the trained codebook

fig3 = plt.figure(num = 1, figsize = (6, 9.5))

ax3 = fig3.add_subplot(211)
ax3.plot(np.array(accs_train).T, '--', marker = 's')
plt.xlabel('Update Steps')
plt.ylabel('Training Accuracy')
plt.title('Scene: {}, number of probing beams: {}'.format(dataset[0], n_wb))

ax3 = fig3.add_subplot(212)
ax3.plot(accs_test, '--', marker = 's')
plt.xlabel('Update Steps')
plt.ylabel('Test Accuracy')
plt.title('Scene: {}, number of probing beams: {}'.format(dataset[-1], n_wb))

plt.show()

## Save the trained codebook

ifsavemodel = True

if ifsavemodel:
    model_savefname = './ZY_dataset/SaveModel/M_{}_{}wb.pt'.format(dataset[0], args.n_wb, noise_power_dBm)
    # model_savefname = './ZY_dataset/SaveModel/M_I3ALL_{}wb.pt'.format(args.n_wb)
    torch.save(net.state_dict(), model_savefname)

## Visualization of the trained codebook

num_codebook = len(trained_codebook)
trained_codebook = np.array(trained_codebook).reshape(num_codebook, args.n_antenna, args.n_wb)
# fig, ax = plot_codebook(trained_codebook[5, :, :].transpose(1, 0))
aaa = trained_codebook[0, :, :]
# fig, ax = plot_codebook(trained_codebook[39, :, [4, 5]].transpose(1, 0))
fig, ax = plot_codebook(aaa.transpose(1, 0))

# trained_codebook.append(net.net.get_codebook())
fig, ax = plot_codebook(temp_codebook)

fig, ax = plot_codebook(beam_weights_init)

fig, ax = plot_codebook(beam_weights_finetuned)

