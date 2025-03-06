import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt


# Function to load multiple variables from a .npz file
def load_variables(file_variables_tuple):
    """Loads the specified variables from a .npz file."""
    file, variable_names = file_variables_tuple
    try:
        with np.load(file) as data:
            return {var: data[var] if var in data else None for var in variable_names}
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return {var: None for var in variable_names}

# Main function to extract multiple variables
def load_variables_from_npz(folder_path, variable_names, num_workers=28, num_files=None, file_selected=None):
    """
    Load multiple variables from .npz files using multiprocessing.

    Returns:
        dict: A dictionary where keys are variable names and values are NumPy arrays.
    """

    # Get list of .npz files
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
   

    if num_files is not None:
        file_list = file_list[:num_files]  # Limit to requested files

    if file_selected is not None:
        # Use list comprehension to select files based on indices in file_selected
        file_list = file_selected

    # Prepare arguments for multiprocessing
    file_variables_tuples = [(file, variable_names) for file in file_list]

    # Use multiprocessing
    num_workers = min(num_workers, os.cpu_count() or 1)
    with Pool(processes=num_workers) as pool:
        data_list = list(tqdm(pool.imap(load_variables, file_variables_tuples), total=len(file_list), desc="Processing files"))

    # Convert list of dictionaries into a dictionary of lists
    data_dict = {var: [] for var in variable_names}
    for entry in data_list:
        for var in variable_names:
            if entry[var] is not None:
                data_dict[var].append(entry[var])

    # Convert lists to NumPy arrays
    for var in data_dict:
        try:
            # If all elements have the same shape, create a standard NumPy array
            sample_shape = set(arr.shape for arr in data_dict[var] if arr is not None)
            if len(sample_shape) == 1:
                data_dict[var] = np.array(data_dict[var], dtype=np.float32)
            else:
                # Store variable-length arrays as dtype=object
                data_dict[var] = np.array(data_dict[var], dtype=object)
        except Exception as e:
            print(f"Error processing variable {var}: {e}")

    return data_dict



def create_masked_dict(data_filter, is_cc, is_nu_e, is_nu_mu, is_nu_tau):
    """
    Filters the data based on the given conditions and returns a masked dictionary.
    
    Parameters:
    data_filter (pd.DataFrame): The input data containing neutrino events.
    is_cc (int): 1 for charged current (CC), 0 for neutral current (NC).
    is_nu_e (int): 1 to filter electron neutrinos (nu_e), 0 otherwise.
    is_nu_mu (int): 1 to filter muon neutrinos (nu_mu), 0 otherwise.
    is_nu_tau (int): 1 to filter tau neutrinos (nu_tau), 0 otherwise.
    
    Returns:
    dict: A dictionary containing filtered "run_number" and "event_id".
    """
    
    if is_cc == 1 and (is_nu_e == 0 and is_nu_mu == 0 and is_nu_tau == 0):
       print("Error: When is_cc is 1, at least one of the neutrino types should be 1.")

    #error if more than one neutrino type is selected
    if is_nu_e + is_nu_mu + is_nu_tau > 1:
        print("Error: Only one neutrino type can be selected.")
        return {}
    
    # --------------


    if is_cc == 1:
        if is_nu_e:
            mask = (data_filter["is_cc"] == 1) & ((data_filter["in_neutrino_pdg"] == 12) | (data_filter["in_neutrino_pdg"] == -12))
        elif is_nu_mu:
            mask = (data_filter["is_cc"] == 1) & ((data_filter["in_neutrino_pdg"] == 14) | (data_filter["in_neutrino_pdg"] == -14))
        elif is_nu_tau:
            mask = (data_filter["is_cc"] == 1) & ((data_filter["in_neutrino_pdg"] == 16) | (data_filter["in_neutrino_pdg"] == -16))
        else:
            mask = (data_filter["is_cc"] == 1)
    else:
        mask = (data_filter["is_cc"] == 0)
    
    return {
        "run_number": data_filter["run_number"][mask],
        "event_id": data_filter["event_id"][mask]
    }