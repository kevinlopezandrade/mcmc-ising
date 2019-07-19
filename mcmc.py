import matplotlib.image as mpimg
import nibabel as nib
import numpy as np

import matplotlib.pylab as plt
from matplotlib.figure import Figure
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from skimage.transform import resize
from scipy.signal import fftconvolve
from sklearn.metrics import hamming_loss

from scipy.optimize import minimize_scalar
from multiprocessing import cpu_count

import warnings
warnings.filterwarnings('ignore')


def flower(size=None, p=0.3):
    """Load image data for denoising
    
    Args:
        size (tuple): image size, default (300,300)
        p (float): noise fraction in [0,1]
        
    Returns:
        noisy: noisy image
        img: ground truth image
    """
    img = mpimg.imread("flower.png")
    img[img > 0.5] = 1
    img[img <= 0.5] = -1
    
    if size is not None:
        img = resize(img, size)
        img[img > 0] = 1
        img[img <= 0] = -1
    
    np.random.seed(13)

    flip = np.random.choice([-1, 1], size=img.shape, p=[p, 1-p])

    noisy = flip * img
    
    return noisy, img


def energy(img_noisy_observation, img_estimate, beta=2.5, mu=1):
    """Compute the energy for given estimate 'img_estimate' which
    is our vector x in the original model, with respect to the 
    observation 'img_noisy_observation', which corresponds to the vector y in the model.

    Args:
        img_estimate (np.ndarray): estimated image matrix
        img_noisy_observation (np.ndarray): noisy image matrix

    Returns:
        energy (float): energy of the estimate given observation
    """
    height = img_estimate.shape[0]
    width = img_estimate.shape[1]

    padded_img_estimate = np.pad(img_estimate, (1,1), mode='constant')
    padded_img_noisy_observation = np.pad(img_noisy_observation, (1,1), mode='constant')
    
    E_first_term = 0
    E_second_term = 0

    for i in range(1, height+1):
        for j in range(1, width+1):
            pixel_value = padded_img_estimate[i,j] 
            noisy_pixel_value = padded_img_noisy_observation[i,j]

            neighbours = [padded_img_estimate[i, j-1],
                          padded_img_estimate[i, j+1],
                          padded_img_estimate[i-1, j],
                          padded_img_estimate[i+1, j]]

            first_term = (pixel_value/4)*sum(neighbours)
            second_term = (pixel_value*noisy_pixel_value)
    
            E_first_term = E_first_term + first_term
            E_second_term = E_second_term + second_term

    E = (-beta/2)*(E_first_term) - (mu)*E_second_term

    return E 


def metropolis(img_noisy_observation, epochs, T=1):
    """Metropolis sampling
    
    For each epoch, loop over every pixel and attempt flip using energy.

    Args:
        img_noisy_observation (np.ndarray): noisy image matrix
        epochs (int): number of rounds over all pixels
        T (float): Temperature of the simulation

    Returns:
        img_estimate (np.ndarray): reconstucted image estimate
        energies (np.ndarray): energy after each epoch
    """
    np.random.seed(7) # Always set the random seed to a lucky number
    
    height = img_noisy_observation.shape[0]
    width = img_noisy_observation.shape[1]
    n_pixels = height * width
    
    noisy_img = img_noisy_observation.copy()
    estimate = img_noisy_observation.copy()
    
    energies = []
    for e in range(epochs):
        
        if e == 0:
            energies.append(energy(noisy_img, estimate))
        
        for cnt, idx in enumerate(np.random.permutation(n_pixels)):
            print("Finished {:6.2f}% of epoch {}".format(cnt/n_pixels * 100, e+1), end="\r")

            row, column = np.unravel_index(idx, dims=(height, width))

            # Compute current energy
            energy_current = energy(noisy_img, estimate)

            # Propose candidate
            flip = -estimate[row, column]

            estimate[row, column] = flip
            energy_candidate = energy(noisy_img, estimate)

            ratio = np.exp((energy_current - energy_candidate)/T)

            u = np.random.uniform()

            acceptance_probability = np.minimum(1, ratio)

            if u < acceptance_probability:
                pass
            else:
                # Undo changes and keep the current state
                unflip = -flip
                estimate[row, column] = unflip


        e = energy(noisy_img, estimate) 
        energies.append(e)
        
    return estimate, np.asarray(energies)


def evaluate_ising(method, img_noisy_observation, img_original, epochs=1, T=1, surpress=False):
    """ Given a sampling method, we will run the sampling procedure 
    for the specifed number of epochs. We measure time and reconstruction
    efficiency.

    Args:
        method (function pointer): sampling method
        img_noisy_observation (np.ndarray): noisy image matrix
        img_original (np.ndarray): original image matrix
        epochs (int): number of epochs to run the sampling, one epoch means going through all pixels once.
        T (float): The positive temperature of the simulation
    """

    start_time = time.time()
    img_estimate, energies = method(img_noisy_observation, epochs, T)
    execution_time = time.time() - start_time

    if not surpress:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

        ax1.plot(energies, marker="x")
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('energy')

        ax2.imshow(img_estimate, cmap='gray')
        ax2.set_title('Reconstruction')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

        ax3.imshow(img_noisy_observation, cmap='gray')
        ax3.set_title('Input')
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)

        plt.suptitle("{} updates per second".format(int(img_original.size*epochs/execution_time)))

        plt.show()
    
    return energies[-1], hamming_loss(y_pred=img_estimate.flatten(), y_true=img_original.flatten())


def distance(coordinates, route=None):
    """Calculate travel distance.
    
    If no route is given, assumes that coordinates are travelled in order,
    starting at the first entry, and connecting the last entry again with the first.
    
    Args:
        coordinates (np.ndarray): route coordinates (assume in units of meters)
        route: permutation of coordinate order
        
    Returns:
        float: traveled distance in units of kilometers
    """
     
    if route is not None:
        coordinates = coordinates[route]
    else:
        coordinates = coordinates[list(range(coordinates.shape[0]))]

    dist = 0
    for coordinate_index in range(coordinates.shape[0]):
        current_point = coordinates[coordinate_index]

        if coordinate_index == coordinates.shape[0] - 1:
            next_point = coordinates[0]
        else:
            next_point = coordinates[coordinate_index + 1]

        distance_two_points = np.linalg.norm(next_point - current_point)

        dist = dist + distance_two_points

    return (dist/1000) # Distance in Km


def distance_permutation(coordinates, before_permutation_route, idx, distance):

    coordinates = coordinates[before_permutation_route]

    N = coordinates.shape[0]

    d1 = np.linalg.norm(coordinates[(idx - 1)%N] - coordinates[idx])
    d2 = np.linalg.norm(coordinates[idx] - coordinates[(idx+1)%N])
    d3 = np.linalg.norm(coordinates[(idx+1)%N] - coordinates[(idx+2)%N])
    
    diff = (d1 + d2 + d3)/1000

    distance_without_involved_permutation = distance - diff


    d1_new = np.linalg.norm(coordinates[(idx - 1)%N] - coordinates[(idx+1)%N])
    d2_new = np.linalg.norm(coordinates[(idx+1)%N] - coordinates[idx])
    d3_new = np.linalg.norm(coordinates[idx] - coordinates[(idx+2)%N])

    new_distance = distance_without_involved_permutation + (d1_new + d2_new + d3_new)/1000

    return new_distance


def evaluate_tsp(method, coordinates, epochs=1):
    """Evaluate sampling method on coordinates
    
    Args:
        method (function pointer): sampling method
        coordinates (np.ndarray): city coordinates, shape Nx2
        epochs (int): number of epochs to run the sampling
    """
    np.random.seed(7)
    N = coordinates.shape[0]
    route = np.random.permutation(N)
 
    start_time = time.time()
    route, distances = method(coordinates, route, epochs)
    execution_time = time.time() - start_time

    if not (np.sort(route) == range(N)).all():
        raise ValueError("Returned route is not valid!")

    x, y = coordinates[route, 0], coordinates[route, 1]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.plot(distances, marker="o", markersize=3)
    plt.plot([0, len(distances)], [27686/1000,27686/1000], c="green") # best known solution
    plt.grid(axis="y")
    
    plt.subplot(122, xticks=[], yticks=[])
    plt.plot(x, y, alpha=0.5, c="blue", marker="o", markerfacecolor="red", markersize=3, linewidth=1)
    plt.plot([x[0], x[-1]], [y[0], y[-1]], alpha=0.5, c="magenta", linewidth=1)
    
    plt.show()


def metropolis_tsp(coordinates, route=None, epochs=1, T=1):
    """Metropolis for TSP

    Args:
        coordinates (np.ndarray): initial route consisting of coordinates
        epochs (int): number of loops through all cities.
        T (float): Temperature of simulation

    Returns:
        route (np.ndarray): optimized route
        distances (np.ndarray): travel distance after each epoch
    """

    np.random.seed(7)
    
    N = coordinates.shape[0]
    
    if route is None:
        route = np.random.permutation(N)
    
    distances = []
    energy_current_state = distance(coordinates, route)
    distances.append(energy_current_state)
    
    for e in range(epochs):
        
        for idx in np.random.permutation(N):
            energy_proposal_state = None

            random_value = route[idx]
            random_value_next = None

            if idx == N-1:
                energy_proposal_state = distance_permutation(coordinates, route, idx, energy_current_state)

                random_value_next = route[0]

                route[idx] = random_value_next
                route[0] = random_value

            else:
                energy_proposal_state = distance_permutation(coordinates, route, idx, energy_current_state)

                random_value_next = route[idx+1]

                route[idx] = random_value_next
                route[idx+1] = random_value

            ratio = np.exp((energy_current_state - energy_proposal_state)/T)

            if np.minimum(ratio, 1) == ratio:

                bernoulli_random_variable = np.random.binomial(1, ratio)

                if bernoulli_random_variable == 0 : # NO jump
                    # Undo the permutation
                    if idx == N-1:
                        route[idx] = random_value
                        route[0] = random_value_next
                    else:
                        route[idx] = random_value
                        route[idx+1] = random_value_next
                else:
                    energy_current_state = energy_proposal_state
            else:
                energy_current_state = energy_proposal_state
        
        dist = distance(coordinates, route)
        distances.append(dist)
        print("epoch {} | Distance : {}".format(e, dist))

    return np.asarray(route), np.asarray(distances)


def simulated_annealing(coordinates, route=None, epochs=1, T=20.0, eta=0.995):
    """Simulated Annealing for TSP

    T(n) = T * eta**n

    Args:
        coordinates (np.ndarray): initial route consisting of coordinates
        route (array): inital route
        epochs (int): number of loops through all cities.
        T (float): Initial temperature
        eta (float): Exponential cooling rate

    Returns:
        route (np.ndarray): optimized route
        distances (np.ndarray): travel distance after each epoch
    """
    np.random.seed(7)
    
    N = coordinates.shape[0]
    
    if route is None:
        route = np.random.permutation(N)
    
    distances = []
    energy_current_state = distance(coordinates, route)

    distances.append(energy_current_state)

    for e in range(epochs):
        for idx in np.random.permutation(N):
            energy_proposal_state = None

            random_value = route[idx]
            random_value_next = route[(idx+1)%N]

            energy_proposal_state = distance_permutation(coordinates, route, idx, energy_current_state)

            route[(idx+1)%N] = random_value
            route[idx] = random_value_next


            acceptance_probability = np.exp((energy_current_state - energy_proposal_state)/T)

            if energy_proposal_state > energy_current_state:

                bernoulli_random_variable = np.random.binomial(1, acceptance_probability)

                if bernoulli_random_variable == 0 : # NO jump
                    # Undo the permutation
                    if idx == N-1:
                        route[idx] = random_value
                        route[0] = random_value_next
                    else:
                        route[idx] = random_value
                        route[idx+1] = random_value_next
                else:
                    energy_current_state = energy_proposal_state
            else:
                energy_current_state = energy_proposal_state

        T = T * eta
        dist = distance(coordinates, route)
        distances.append(dist)
        print("epoch {} | Distance : {} | T : {}".format(e, dist, T))

    return np.asarray(route), np.asarray(distances)




def local_energy_change(noisy, estimate, i, j, beta, mu):
    """
    
    Local energy difference between unflipped and flipped pixel i,j 
    
    Args:
        noisy: noisy reference image
        estimate: current denoising estimate
        i,j: Position of pixel
    
    Returns:
        float: local energy difference when pixel i,j is flipped 
    """

    neighbours = [estimate[i, j-1],
                  estimate[i, j+1],
                  estimate[i-1, j],
                  estimate[i+1, j]]

    y = noisy[i, j]
    
    x_r = estimate[i,j]

    # dE = (-beta/2)* x_r * sum(neighbours) - (2 * mu * x_r * y)
    dE = (-beta/2)* x_r * sum(neighbours) - (2 * mu * x_r * y)

    return dE


def local_metropolis(img_noisy_observation, epochs, T=1, beta=2.5, mu=1):
    """Metropolis sampling
    
    For each epoch, loop over every pixel and attempt flip using local_energy_change 

    Args:
        img_noisy_observation (np.ndarray): noisy image matrix
        epochs (int): number of rounds over all pixels
        T (float): Temperature of simulation

    Returns:
        img_estimate (np.ndarray): reconstucted image estimate
        energies (np.ndarray): energy after each epoch
    """
    np.random.seed(7) # Always set the random seed to a lucky number
    
    height = img_noisy_observation.shape[0]
    width = img_noisy_observation.shape[1]
    n_pixels = height * width
    
    noisy_img = img_noisy_observation.copy()
    estimate = img_noisy_observation.copy()

    padded_img_estimate = np.pad(estimate, (1,1), mode='constant')
    padded_img_noisy = np.pad(noisy_img, (1,1), mode='constant')

    energies = []
    for e in range(epochs):
        
        if e == 0:
            energies.append(energy(padded_img_noisy[1:-1, 1:-1], padded_img_estimate[1:-1, 1:-1]))
        
        for cnt, pix in enumerate(np.random.permutation(n_pixels)):
            
            row, column = np.unravel_index(pix, dims=(height, width))
            row = row + 1
            column = column + 1

            dE = local_energy_change(padded_img_noisy, padded_img_estimate, row, column, beta, mu)
            
            ratio = np.exp(dE)

            u = np.random.uniform()

            acceptance_probability = np.minimum(1, ratio)

            if u < acceptance_probability:
                flip = -padded_img_estimate[row, column]
                padded_img_estimate[row, column] = flip
            else:
                pass
            
        print("Finished epoch {}".format(e+1), end="\r")
        energies.append(energy(padded_img_noisy[1:-1, 1:-1], padded_img_estimate[1:-1, 1:-1]))
        
    return padded_img_estimate[1:-1, 1:-1], np.asarray(energies)

def comparasion_plot(avg_img_array, noise_img_array, index=0):

    fig = plt.figure()
    fig.suptitle("Testing")

    axes_array = fig.subplots(1,2)

    slice1 = avg_img_array[:,:,index]
    slice2 = noise_img_array[:,:,index]

    axes_array[0].imshow(slice1, cmap='gray')
    axes_array[0].set_title("Averaged axial image")

    axes_array[1].imshow(slice2, cmap='gray')
    axes_array[1].set_title("Noise axial image")

    fig.show()
