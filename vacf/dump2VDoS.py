'''
VDoS - Vibrational density of states
This script processes LAMMPS MD trajectories containing velocity data to
estimate the vibrational density of states (VDoS). The input format should
include the following fields:
ID TYPE x y z vx vy vz.
'''

import time
import sys
import argparse
import numpy as np
import concurrent.futures
import itertools
import glob

# Finds and returns the index of the first True condition in a Boolean list
def first_occurrence(cond):
    for i, c in enumerate(cond):
        if c:
            return i

# Returns the number of lines in a file
def file_len(fname):
    with open(fname, 'r') as f:
        for i, _ in enumerate(f):
            pass
        return i + 1

'''
    Function to estimate the ensemble average of the velocity auto-correlation
    for a given run.

    Inputs:
    - fname: Filename of the dump file (must be velocities*.dump)
    - mass_by_type: List of atomic masses for each atom type
    - path: Directory path where dump files are stored
    - output_freq: Frequency of progress updates during processing

    Outputs:
    - intensity: VDoS intensity as a function of frequency
    - n_atom: number of atoms in the simulation
    - n_step: number of timesteps in the simulation
'''
def get_VDoS(fname, mass_by_type, path, output_freq):
    n_dim = 3 # 3D space (x, y, z)
    # Maximum allowed size for NumPy arrays (for memory management)
    max_mem_size = 2e10

    # Extract the file label from the filename to identify runs
    prefix = path + 'velocities'
    suffix = '.dump'
    f_label = fname
    if fname.startswith(prefix):
        f_label = f_label[len(prefix):]
    else:
        raise ValueError(
                f'fname ({fname}) does not start with expected prefix.')

    if fname.endswith(suffix):
        f_label = f_label[:-len(suffix)]
    else:
        raise ValueError(f'fname ({fname}) does not end with expected suffix.')

    start_time = time.time() # Record start time for performance tracking
    print(f'Processing file: {f_label}, time = {start_time}')

    '''
        Read the number of lines in the first dump file for determining
        timestep counts
    '''
    n_header_line = 9 # Fixed header size in the LAMMPS dump file
    n_line        = file_len(flist[0])

    with open(fname, 'r') as f:
        # Skip the header lines to reach atom data
        for i in range(3):
            f.readline()
        n_atom = int(f.readline()) # Number of atoms
        if f_label == '1': # Only print this for the first file
            print(f'number of atoms = {n_atom}')
        for i in range(4):
            f.readline()

        # Parse the atom data (IDs, types, and velocities)
        header = f.readline()
        header = header[len('ITEM: ATOMS '):].split() # Extract column names
        header = np.asarray(header, dtype = 'object')

        id_col = np.where(header == 'id')[0][0] # Column index for atom IDs
        # Column index for atom types
        type_col = np.where(header == 'type')[0][0]
        # Columns for velocities
        v_cols = np.where([x in ['vx', 'vy', 'vz'] for x in header])[0]

        # Calculate the number of timesteps in the trajectory
        n_step = int(n_line/(n_header_line + n_atom))
        # Initialize an array for atom types
        types = np.zeros(n_atom, dtype = np.int64)

        '''
            check if required size of array is too large for memory:
               if so, create a memmap array;
               if not, create a normal array
        '''
        vel_shape = (n_step, n_dim*n_atom)
        if sys.getsizeof(np.float64)*n_step*n_dim*n_atom > max_mem_size:
            vel_name = 'vel_array' + f_label + '.memmap'
            # Memory-mapped array
            vel_array = np.memmap(vel_name,
                                  mode  = 'w+',
                                  dtype = np.float64,
                                  shape = vel_shape)
        else:
            vel_array = np.zeros(vel_shape) # Regular in-memory array

        # Intensity spectrum to be calculated
        intensity = np.zeros(n_step//2 + 1)

        # Array for storing atom IDs
        id_list = np.zeros(n_atom, dtype = np.int64)

        # Process the atom information for each timestep
        for na in range(n_atom):
            line = f.readline().split()
            id_list[na] = np.int64(line[id_col])
            types[na] = np.int64(line[type_col])

    # Sort atom IDs for efficient access during velocity extraction
    sorted_inds = id_list.argsort()
    id_list = id_list[sorted_inds]
    types = types[sorted_inds]

    # Open the dump file again for velocity data extraction
    with open(fname, 'r') as f:
        for t in range(n_step): # Loop through all timesteps

            # Skip header lines for each timestep
            for _ in range(n_header_line):
                f.readline()

            # Extract velocity data for each atom
            for na in range(n_atom):
                # Read string -> strip '\n' char -> split into new list
                line = f.readline().split()
                # Find the atom's index by its ID
                i = first_occurrence(id_list == int(line[id_col]))
                # Collect the velocity components
                vel = np.array( [float(line[i]) for i in v_cols] )
                # Store the velocity data
                vel_array[t, n_dim*i:n_dim*(i + 1)] = vel

            # Print progress at intervals if output_freq is set
            if output_freq and (not (t % output_freq)):
                elapsed = time.time() - start_time
                print(f'f = {f_label}, t = {t}, elapsed = {elapsed} s')

    unique_types = np.unique(types) # Find unique atom types for mass lookup

    # Process velocities for each atom to calculate the intensity spectrum
    for atom_id in range(n_atom):
        # Velocity time series for this atom
        v = vel_array[:, n_dim*atom_id:n_dim*(atom_id + 1)]
        # Mass of the atom's type
        mi = mass_by_type[first_occurrence( unique_types == types[atom_id] )]
        # Fourier transform of the velocity
        fv = np.fft.rfft(v, axis = 0)
        # Power spectrum (magnitude squared)
        M2 = np.real(fv)**2 + np.imag(fv)**2
        # Accumulate the intensity for this atom
        intensity += mi*np.sum(M2, axis = 1)

    return (intensity, n_atom, n_step)
    
# pickleable function for parallelization
def get_VDoS_wrapper(fname, args):
    return get_VDoS(fname, args.mass_by_type, args.path, args.output_freq)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument('--dt',
                        type = float,
                        default = 1.6E-15,
                        help = 'Timestep value [s]')
    parser.add_argument('--path',
                        type = str,
                        default = './',
                        help = 'Path with dump files')
    parser.add_argument('--mass_by_type',
                        type = float,
                        nargs = '+',
                        default =
                         [12.0107, 12.0107, 15.999, 1.00784, 1.00784, 1.00784],
                        help = 'Atomic mass for each atom type [amu]')
    parser.add_argument('--output_freq',
                        type = int,
                        default = 1000,
                        help =
                         'Controls frequency of updates (0 for never)')

    args = parser.parse_args()
    args.mass_by_type = np.array(args.mass_by_type)

    n_dim = 3 # 3D space (x, y, z)
    Hz_to_inv_cm = 1E-2/299792458 # conversion constant for frequency units

    flist = glob.glob(args.path + 'velocities*.dump')

    intensity_list = []
    n_atom = 0
    n_step = 0

    # process files in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(get_VDoS_wrapper, flist[0:], \
                                   itertools.repeat(args)):
            intensity_list.append(result[0])
            n_atom += result[1]
            if n_step == 0:
                n_step = result[2]

    nu = np.fft.rfftfreq(n_step, d = args.dt) # wavenumber [Hz]
    wavenumber = nu*Hz_to_inv_cm # converts wavenumber to cm^{-1}

    intensity = intensity_list[0] # intensity array from first file
    for i in range(1, len(intensity_list)):
        intensity += intensity_list[i]

    # normalize the intensity spectrum to integrate to n_dim*n_atom
    intensity *= n_dim*n_atom/ \
                       ( np.sum(intensity)*np.mean(np.diff(wavenumber)) )

    np.save('wavenumber', wavenumber)
    np.save('intensity', intensity)

    print('complete')