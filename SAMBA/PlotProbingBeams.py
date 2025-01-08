import torch
import numpy as np
import matplotlib.pyplot as plt

# theta: n_wb x n_antenna
def Plot_BeamPattern(theta):
    real_kernel = (1/8) * torch.cos(theta)
    imag_kernel = (1/8) * torch.sin(theta)
    beam_weights = real_kernel + 1j*imag_kernel

    beam_weights = beam_weights.cpu().detach().clone().numpy()

    temp_codebook = np.array*(beam_weights)

    fig = plt.figure()
    ax = fig.add_subplot(111, polar = True)
    for _, beam in enumerate(temp_codebook):
        phi, bf_gain = calc_beam_pattern(beam)
        ax.plot(phi, bf_gain)
    ax.grid(True)
    ax.set_rlabel_position(-90)  # Move radial labels away from plotted line
    return fig, ax

def calc_beam_pattern(beam, resolution = int(1e3), n_antenna = 64, k = 0.5):
    phi_all = np.linspace(-np.pi/2,np.pi/2,resolution)
    array_response_vectors = np.tile(phi_all, (n_antenna, 1)).T
    array_response_vectors = -1j*2*np.pi*k*np.sin(array_response_vectors)
    array_response_vectors = array_response_vectors * np.arange(n_antenna)
    array_response_vectors = np.exp(array_response_vectors)/np.sqrt(n_antenna)
    gains = abs(array_response_vectors.conj() @ beam)**2
    return phi_all, gains