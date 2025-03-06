import csv
import numpy as np
from Utils.event_filter import load_variables_from_npz



def save_ID_event(folder_path, features_to_load):
    data = load_variables_from_npz(folder_path, features_to_load)

    cc_true = data["is_cc"] == 1
    cc_false = data["is_cc"] == 0

    nu_tau = data["in_neutrino_pdg"] == 16 
    nu_mu = data["in_neutrino_pdg"] == 14
    nu_e = data["in_neutrino_pdg"] == 12

    nc = cc_false
    cc_nu_e = cc_true & nu_e
    cc_nu_mu = cc_true & nu_mu
    cc_nu_tau = cc_true & nu_tau

    run_number_nc = data["run_number"][nc]
    event_id_nc = data["event_id"][nc]

    run_number_e = data["run_number"][cc_nu_e]
    event_id_e = data["event_id"][cc_nu_e]

    run_number_mu = data["run_number"][cc_nu_mu]
    event_id_mu = data["event_id"][cc_nu_mu]

    run_number_tau = data["run_number"][cc_nu_tau]
    event_id_tau = data["event_id"][cc_nu_tau]

    # Save the data to a CSV file
    with open('id_events.txt', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')

        # Write the header (column names)
        writer.writerow(['Run_e', 'EventID_e', 'Run_mu', 'EventID_mu', 'Run_tau', 'EventID_tau'])

        # Make sure all arrays are the same length (use zip to align them)
        max_length = max(len(run_number_e), len(run_number_mu), len(run_number_tau), len(event_id_nc))
        for i in range(max_length):
            run_e = run_number_e[i] if i < len(run_number_e) else "000"
            event_e = event_id_e[i] if i < len(event_id_e) else "000"

            run_mu = run_number_mu[i] if i < len(run_number_mu) else "000"
            event_mu = event_id_mu[i] if i < len(event_id_mu) else "000"

            run_tau = run_number_tau[i] if i < len(run_number_tau) else "000"
            event_tau = event_id_tau[i] if i < len(event_id_tau) else "000"

            run_nc = run_number_nc[i] if i < len(run_number_nc) else "000"
            event_nc = event_id_nc[i] if i < len(event_id_nc) else "000"

            # Write the row with space-separated values
            writer.writerow([int(run_e), int(event_e), int(run_mu), int(event_mu), int(run_tau), int(event_tau), int(run_nc), int(event_nc)])

    print("File 'id_events.txt' has been saved.")




def get_event_ID(is_cc, is_nu_e, is_nu_mu, is_nu_tau, filename='id_events.txt'):
    neutrino_map = {
        "NC": 0,
        "e": 1,
        "mu": 2,
        "tau": 3
    }
    
    neutrino_type = "e" if is_cc and is_nu_e else \
                    "mu" if is_cc and is_nu_mu else \
                    "tau" if is_cc and is_nu_tau else "NC"

    
    events_id = []
    with open(filename, 'r') as f:
        next(f)  # Skip the header
        
        for line in f:
            parts = line.strip().split()
            
            if len(parts) == 8:
                run_nc, event_nc, run_e, event_e, run_mu, event_mu, run_tau, event_tau = map(int, parts)
                events_id.append((run_nc, event_nc, run_e, event_e, run_mu, event_mu, run_tau, event_tau))
            else:
                print(f"Skipping malformed line: {line.strip()}")
    
    index = neutrino_map[neutrino_type] * 2  # Get index corresponding to neutrino type
    
    file_list = [f"{event[index]}_{event[index + 1]}.npz" if event[index] != 0 else None for event in events_id]
    
    return [f for f in file_list if f is not None]


# MORE EASY VERSION 

import numpy as np

def get_ID(folder_path, is_cc, is_nu_e, is_nu_mu, is_nu_tau):
    """
    Loads neutrino interaction data from an NPZ file and filters events based on user-specified conditions.

    Args:
        folder_path (str): Path to the folder containing the NPZ file.
        is_cc (int, optional): 1 for Charged Current (CC), 0 for Neutral Current (NC). Default is 1.
        is_nu_e (int, optional): 1 to filter electron neutrino interactions, 0 otherwise. Default is 0.
        is_nu_mu (int, optional): 1 to filter muon neutrino interactions, 0 otherwise. Default is 0.
        is_nu_tau (int, optional): 1 to filter tau neutrino interactions, 0 otherwise. Default is 1.

    Returns:
        dict: Dictionary containing filtered run numbers and event IDs.
    """
    # Define the features to load| Minimum number of features for ID
    features_to_load = ["run_number", "event_id", "is_cc", "in_neutrino_pdg", "out_lepton_pdg"]

    # Load data
    data = load_variables_from_npz(folder_path, features_to_load)

    # Filter CC or NC events
    cc_filter = (data["is_cc"] == is_cc)

    # Filter neutrino types based on user selection
    nu_e_filter = (data["in_neutrino_pdg"] == 12) | (data["in_neutrino_pdg"] == -12) 
    nu_mu_filter = (data["in_neutrino_pdg"] == 14) | (data["in_neutrino_pdg"] == -14)
    nu_tau_filter = (data["in_neutrino_pdg"] == 16) | (data["in_neutrino_pdg"] == -16) 

    # Combine the filters
    event_filter = cc_filter & (nu_e_filter | nu_mu_filter | nu_tau_filter)

    # Extract run numbers and event IDs based on the filters
    return {
        "run_number": data["run_number"][event_filter],
        "event_id": data["event_id"][event_filter]
    }
