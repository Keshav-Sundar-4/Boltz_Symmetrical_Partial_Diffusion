# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers for euclidean symmetry group operations 
Updates from Yang and Rian regarding the I symmetry and Radius 20240925
This module contains pytorch layers for symmetry operations for point groups (Cyclic, Dihedral, Tetrahedral, Octahedral and Icosahedral)
"""

import itertools
import math
from itertools import product
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from tqdm import tqdm

TAU = 0.5 * (1 + math.sqrt(5))

ROT_DICT = {
    "O": [
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
    ],
    "T": [
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    ],
}


def get_point_group(group: str) -> torch.Tensor:
    """get representation of group elements as torch.Tensor

    Args:
        group (str): group names, selecting from {"C_{n}" , "D_{n}", "T", "O", "I" }

    Returns:
        torch.Tensor: rotation matrices for queried point groups
    """
    if group.startswith("C"):
        n = group.split("_")[1]
        G = get_Cn_groups(int(n))
    elif group.startswith("D"):
        n = group.split("_")[1]
        G = get_Dn_groups(int(n))
    elif group == "I":
        # *Use the corrected function to get the full group*
        # The pseudoquotient logic is handled *separately* when needed,
        # e.g., during symmetry setup in diffusion.py.
        # get_point_group("I") should return the full 60 rotations.
        G = get_I_rotations_full()
    elif group == "O" or group == "T":
        # Convert ROT_DICT to tensors if not already done
        if isinstance(ROT_DICT[group], list):
             G = torch.tensor(ROT_DICT[group], dtype=torch.float32)
        else: # Assume it's already a tensor
             G = ROT_DICT[group].float()
    else:
        raise ValueError(f"Point group '{group}' not available")

    return G.float() # Ensure float type



def get_Cn_groups(n: int) -> torch.Tensor:
    """get rotation matrices for Cyclic groups

    Args:
        n (int): symmetry order

    Returns:
        torch.Tensor: n x 3 x 3
    """
    G = []
    for ri in range(n):
        # Use double precision for intermediate calculations then round
        angle = ri * 2.0 * np.pi / n
        cos_phi = np.cos(angle)
        sin_phi = np.sin(angle)

        g = np.array(
            [[cos_phi, -sin_phi, 0.0], [sin_phi, cos_phi, 0.0], [0.0, 0.0, 1.0]]
        )
        # Round to a reasonable number of decimal places to ensure consistency
        G.append(np.round(g, 8)) 

    return torch.Tensor(np.array(G))


def get_Dn_groups(n: int) -> torch.Tensor:
    """get rotation matrices for Dihedral groups

    Args:
        n (int): symmetry order

    Returns:
        torch.Tensor: 2n x 3 x 3
    """
    # Use double precision for intermediate calculations
    angle = 2.0 * np.pi / n
    cos_phi = np.cos(angle)
    sin_phi = np.sin(angle)

    rot_generator = np.array(
        [[cos_phi, -sin_phi, 0.0], [sin_phi, cos_phi, 0.0], [0.0, 0.0, 1.0]]
    )
    
    # Round the generator matrix to ensure consistency
    rot_generator = np.round(rot_generator, 8)

    b = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    b = np.round(b, 8) # Ensure b is also rounded consistently

    G = []
    # Start with identity matrix equivalent (rotation by 0)
    current_rot = np.identity(3) 
    
    for _ in range(n):
        # Append the C_n rotation element
        G.append(current_rot) 
        # Append the D_n reflection element (b composed with current rotation)
        G.append(b @ current_rot) 
        # Apply the next rotation
        current_rot = current_rot @ rot_generator
        # Round after multiplication to prevent accumulating errors
        current_rot = np.round(current_rot, 8) 

    # Convert the list of numpy arrays to a single torch tensor
    return torch.Tensor(np.array(G))


def get_I_rotations_full() -> torch.Tensor:
    """
    Generates the full set of 60 rotation matrices for the Icosahedral group (I).
    Uses a generator-based approach with tolerance-based uniqueness check.
    """
    # Define generators (ensure they are float tensors)
    g1 = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32) # C2
    g2 = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32) # C3
    g3 = torch.tensor( # C5 related (ensure float)
        [
            [0.5, -0.5 * TAU, 0.5 / TAU],
            [0.5 * TAU, 0.5 / TAU, -0.5],
            [0.5 / TAU, 0.5, 0.5 * TAU],
        ], dtype=torch.float32
    )
    identity = torch.eye(3, dtype=torch.float32)
    generators = [g1, g2, g3] # Only non-identity needed for multiplication step

    # Store generated ops in a list, check uniqueness with allclose
    sym_ops_list = [identity]
    newly_added = [identity]
    
    # Set a tolerance for uniqueness checks
    tolerance = 1e-5

    max_iterations = 10 # Safety break
    iterations = 0

    while newly_added and iterations < max_iterations:
        iterations += 1
        current_generation = []
        
        # Multiply newly added elements by all generators
        for mat1 in newly_added:
            for gen in generators:
                prod = mat1 @ gen

                # Check if prod is close to any existing matrix in sym_ops_list
                is_new = True
                for existing_op in sym_ops_list:
                    if torch.allclose(prod, existing_op, atol=tolerance):
                        is_new = False
                        break
                
                if is_new:
                    current_generation.append(prod)
                    # Add to main list immediately to avoid adding duplicates within the same generation
                    sym_ops_list.append(prod) 
        
        # Update newly_added for the next iteration - filter duplicates within current_generation again
        # This secondary check might be redundant now but adds robustness
        unique_in_generation = []
        for mat_new in current_generation:
             is_truly_new = True
             for existing_op in unique_in_generation: # Check against others found *in this pass*
                 if torch.allclose(mat_new, existing_op, atol=tolerance):
                      is_truly_new = False
                      break
             if is_truly_new:
                 unique_in_generation.append(mat_new)

        newly_added = unique_in_generation

        # Optional: Check size for early exit or warning
        # if len(sym_ops_list) > 60:
        #      print(f"Warning: Exceeded 60 ops during generation ({len(sym_ops_list)}). Check tolerance/logic.")
        #      break # Or let it finish and rely on final count check

    final_ops = torch.stack(sym_ops_list, dim=0)

    # Final check for exact count
    if final_ops.shape[0] != 60:
         print(f"Warning: Generated {final_ops.shape[0]} unique rotations for I, expected 60. Check tolerance ({tolerance}) or generators.")
         # Optional: Add a final unique check with tolerance, though the loop should handle it
         # final_ops = torch.unique(torch.round(final_ops, decimals=5), dim=0) # Less reliable fallback
         # print(f"Post-unique check size: {final_ops.shape[0]}")


    return final_ops.float() # Ensure correct dtype



def get_pseudoquotient(
    G_name: str,
    G_div_name: str
) -> torch.Tensor:
    """Find the set of rotation matrices representing the pseudoquotient G / G_div.

    Calculates a set of representative rotation matrices such that when each representative
    is composed with all elements of the subgroup G_div, the full group G is generated,
    with each element of G generated exactly once.

    Args:
        G_name (str) : group name, selecting from {"C_{n}" , "D_{n}", "T", "O", "I" }
        G_div_name (str) : divisor subgroup name, must be {"C_{n}"} for some n.

    Returns:
        G_quot (torch.Tensor) : matrices representing the pseudoquotient G / G_div.
                                Shape (order(G)/order(G_div), 3, 3).
    """
    if not G_div_name.startswith("C_"):
        raise ValueError(f"Divisor group {G_div_name} must be a Cyclic group (C_n).")

    n_div = int(G_div_name.split('_')[1])

    # Get the full rotation sets for G and G_div
    if G_name == 'I':
        # Ensure it uses the corrected generator function
        G_tensor = get_I_rotations_full()
    else:
        G_tensor = get_point_group(G_name) # This needs get_point_group defined above get_pseudoquotient

    G_div_tensor = get_Cn_groups(n_div) # Directly get C_n group

    # Move tensors to the same device before calculations if needed (usually CPU is fine here)
    # G_tensor = G_tensor.to(device)
    # G_div_tensor = G_div_tensor.to(device)

    order_G = G_tensor.shape[0]
    order_G_div = G_div_tensor.shape[0]

    # --- CRITICAL CHECK ---
    if order_G % order_G_div != 0:
        # Give more informative error if G generation failed
        if G_name == 'I' and order_G != 60:
             raise ValueError(f"Failed to generate the full Icosahedral group (got {order_G} elements, expected 60). Cannot compute I/C3 pseudoquotient.")
        else:
             raise ValueError(f"Order of G ({G_name}, order={order_G}) must be divisible by order of G_div ({G_div_name}, order={order_G_div}).")

    num_quotient_elements = order_G // order_G_div

    # Use tolerance for floating point comparisons when checking coverage
    tol = 1e-5

    # Keep track of covered elements using indices from G_tensor for efficiency
    covered_indices = set()
    representatives = [] # Store the actual tensors

    # Iterate through G to find representatives
    for idx, g in enumerate(G_tensor):

        # If this element's index is already covered, skip
        if idx in covered_indices:
            continue

        # This element g is a potential new representative
        representatives.append(g)

        # Find all elements in the coset g * G_div and mark their indices in G_tensor as covered
        for g_div in G_div_tensor:
            coset_element = g @ g_div

            # Find which element in G_tensor this coset_element corresponds to (using tolerance)
            found_match = False
            for g_idx, g_orig in enumerate(G_tensor):
                 # Optimization: skip if already covered
                 # if g_idx in covered_indices: continue # This might slow down more than help

                 if torch.allclose(coset_element, g_orig, atol=tol):
                      covered_indices.add(g_idx)
                      found_match = True
                      break # Found the match for this coset element

            # This warning indicates an issue either in group closure or tolerance
            # if not found_match:
            #      print(f"Warning: Generated coset element could not be matched back to an element in G (tolerance {tol}).")

        # Stop once we have found enough representatives
        if len(representatives) == num_quotient_elements:
            break

    if len(representatives) != num_quotient_elements:
         raise RuntimeError(f"Failed to find the correct number of quotient representatives. "
                           f"Expected {num_quotient_elements}, found {len(representatives)}. "
                           f"(Covered {len(covered_indices)} elements out of {order_G})")

    # Sanity check: ensure all elements of G were covered
    if len(covered_indices) != order_G:
         print(f"Warning: Cosets did not perfectly cover G. "
               f"Expected {order_G} elements covered, found {len(covered_indices)}. Check tolerance ({tol}) or group generation.")


    return torch.stack(representatives, dim=0).float()



def subsample(
    X: torch.Tensor,
    C: torch.Tensor,
    G: torch.Tensor,
    knbr: int,
    seed_idx: Optional[int] = None,
):
    """generate substructures based on distances between subunit COM

    Args:
        X (torch.Tensor): structures (batch, num_residues_total, atoms, 3),
                          assumes residues are ordered correctly by subunit.
        C (torch.Tensor): chain map (batch, num_residues_total)
        G (torch.Tensor): rotation matrices (num_subunits, 3, 3) used to generate X
        knbr (int): number of nearest neighbors (subunits) to include *besides* the seed.
        seed_idx (int, optional): Index (0 to num_subunits-1) of the seed subunit.
                                  Randomly selected if None. Defaults to None.

    Returns:
        tuple: substructure coordinates (X_subdomain), chain map (C_subdomain),
               indices of the selected subunits (subdomain_idx), seed subunit index (seed_idx)
    """
    num_subunits = G.shape[0]
    batch_size, num_residues_total, num_atoms, _ = X.shape
    
    if num_residues_total % num_subunits != 0:
        raise ValueError("Total number of residues must be divisible by the number of subunits.")
    residues_per_subunit = num_residues_total // num_subunits

    # Ensure knbr is valid
    if knbr >= num_subunits:
       print(f"Warning: knbr ({knbr}) >= number of subunits ({num_subunits}). Selecting all subunits.")
       knbr = num_subunits - 1
    if knbr < 0:
        raise ValueError("knbr cannot be negative.")

    # Reshape X to easily access subunits: (batch, num_subunits, residues_per_subunit, atoms, 3)
    X_reshaped = X.reshape(batch_size, num_subunits, residues_per_subunit, num_atoms, 3)

    # Calculate Center of Mass (COM) for each subunit
    # Assuming atoms have roughly equal mass, use mean position. Use C-alpha (atom index 1) for COM calc.
    X_chain_com = X_reshaped[:, :, :, 1, :].mean(dim=2) # Shape: (batch, num_subunits, 3)

    # Select seed subunit index if not provided
    if seed_idx is None:
        # Select a random seed for each batch element if batch_size > 1
        # For now, assume batch_size = 1 as in original logic
        if batch_size > 1:
             raise NotImplementedError("Batch size > 1 not fully handled for random seed selection in subsample.")
        seed_idx = torch.randint(0, num_subunits, (1,), device=X.device).item()
    elif not (0 <= seed_idx < num_subunits):
         raise ValueError(f"seed_idx ({seed_idx}) out of range [0, {num_subunits-1}]")


    # Calculate distances between subunit COMs (for the first batch element if batch_size > 1)
    # Shape: (num_subunits, num_subunits)
    Dis_chain = torch.cdist(X_chain_com[0], X_chain_com[0], p=2)

    # Find the indices of the (knbr + 1) nearest subunits to the seed subunit
    # Include the seed itself (distance 0)
    _, subdomain_idx = torch.topk(Dis_chain[seed_idx], k=knbr + 1, largest=False, sorted=True)
    # subdomain_idx shape: (knbr + 1)

    # Select the corresponding subunits from X
    # X_subdomain shape: (batch_size, knbr + 1, residues_per_subunit, num_atoms, 3)
    X_subdomain_b = X_reshaped[:, subdomain_idx, :, :, :]
    
    # Reshape back to standard format: (batch_size, (knbr+1)*residues_per_subunit, atoms, 3)
    X_subdomain = X_subdomain_b.reshape(batch_size, -1, num_atoms, 3)

    # Select and reshape the chain map C
    # C shape: (batch_size, num_residues_total)
    C_reshaped = C.reshape(batch_size, num_subunits, residues_per_subunit)
    C_subdomain_b = C_reshaped[:, subdomain_idx, :]
    # C_subdomain shape: (batch_size, (knbr+1)*residues_per_subunit)
    C_subdomain = C_subdomain_b.reshape(batch_size, -1)

    return X_subdomain, C_subdomain, subdomain_idx, seed_idx


def symmetrize_XCS(
    X: torch.Tensor,
    C: torch.LongTensor,
    S: torch.LongTensor,
    G: torch.Tensor,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """Symmetrize a protein structure (monomer) with a given symmetry group.

    Takes the coordinates (X), chain IDs (C), and sequence (S) of a single
    asymmetric unit (monomer) and generates the full symmetric complex by applying
    the rotation matrices in G.

    Args:
        X (torch.Tensor): Monomer coordinates tensor with shape
                          `(batch_size, num_residues, num_atoms, 3)`.
                          Typically num_atoms=4 for N, CA, C, O.
        C (torch.LongTensor): Monomer chain tensor with shape `(batch_size, num_residues)`.
                             All values should ideally be the same (e.g., 0 or 1) for a monomer.
        S (torch.LongTensor): Monomer sequence tensor with shape `(batch_size, num_residues)`.
        G (torch.Tensor): Symmetry group rotation matrices tensor with shape `(n_sym, 3, 3)`.
        device (str or torch.device, optional): The device for output tensors. Defaults to "cpu".

    Returns:
        Tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
            - X_complex (torch.Tensor): Symmetrized coordinates tensor with shape
                `(batch_size, num_residues * n_sym, num_atoms, 3)`.
            - C_complex (torch.LongTensor): Modified chain tensor for the complex with shape
                `(batch_size, num_residues * n_sym)`. Chains are typically numbered 1 to n_sym.
            - S_complex (torch.LongTensor): Replicated sequence tensor for the complex with shape
                `(batch_size, num_residues * n_sym)`.
    """
    n_sym = G.shape[0]
    batch_size, num_residues, num_atoms, _ = X.shape

    # Move G to the same device as X for the einsum operation
    G = G.to(X.device)

    # Apply rotations:
    # G is (n_sym, 3, 3) -> (g, i, j) where i is output coord, j is input coord
    # X is (batch, n_res, n_atoms, 3) -> (b, n, l, j) where l is atom index, j is input coord
    # We want result: (batch, n_sym, n_res, n_atoms, 3) -> (b, g, n, l, i)
    # einsum: 'gij,bnlj->bgnli' seems correct
    X_complex_bgnli = torch.einsum('gij,bnlj->bgnli', G, X)

    # Reshape to combine n_sym and n_res dimensions: (batch, n_sym * n_res, n_atoms, 3)
    X_complex = X_complex_bgnli.reshape(batch_size, n_sym * num_residues, num_atoms, 3).to(device)

    # Generate new chain IDs for the complex.
    # Assign a unique chain ID for each subunit generated by G.
    # Chain IDs will range from 1 to n_sym.
    # Assumes original C contains a single chain ID (e.g., all 0s or 1s).
    # We create IDs [1, 1, ..., 1, 2, 2, ..., 2, ..., n_sym, ..., n_sym]
    chain_ids_per_subunit = torch.arange(1, n_sym + 1, device=X.device) # Shape: (n_sym)
    # Repeat each chain ID 'num_residues' times
    C_complex = chain_ids_per_subunit.repeat_interleave(num_residues) # Shape: (n_sym * num_residues)
    # Expand for batch dimension
    C_complex = C_complex.unsqueeze(0).expand(batch_size, -1).to(device) # Shape: (batch, n_sym * num_residues)


    # Replicate the sequence for each subunit
    S_complex = S.repeat(1, n_sym).to(device) # Shape: (batch, n_sym * num_residues)

    return X_complex, C_complex, S_complex

# --- Functions from the original file that were requested to be kept unchanged ---
# (calculate_radius_from_mw, get_atom_indices_for_chain, calculate_ring_radius, 
#  get_subunit_atom_indices, reorder_point_group)
# Note: These functions were present in the first `symmetry_awareness.py` version provided
# in the prompt, but *not* in the second version. Based on the final instruction
# "Output the full new python file" and the goal to only replace 'I' logic, these
# functions from the *first* version should be included if they were intended to be part
# of the baseline file. We include them here as they were in the initial context.

def calculate_radius_from_mw(molecular_weight: float, n: int, symmetry_type: str) -> float:
    """Calculates the radius of the COMPLEX for COM regularization based on 
    molecular weight and symmetry type.

    The radius is defined as the distance from the center of the complex to the 
    center of mass (COM) of a constituent protein.

    Args:
        molecular_weight: Molecular weight of a SINGLE MONOMER.
        n: Symmetry order (e.g., number of monomers for Cn symmetry).
                  For T, O, I, n is 12, 24, 60 respectively. For D_n it's 2n.
        symmetry_type: Type of symmetry (e.g., "C_n", "D_n", "T", "O", "I").

    Returns:
        The calculated radius of the complex.
    """

    # Estimate the radius of a single monomer (assuming it's roughly spherical).
    # This formula is a common approximation (Rg ~ 0.66 * MW^(1/3))
    # The division by 1.3 seems like an empirical adjustment factor.
    r_monomer = 0.66 * (molecular_weight ** (1/3))
    r_monomer = r_monomer/1.3 

    if symmetry_type.startswith("C"):
        order_n = int(symmetry_type.split('_')[1])
        if order_n != n:
             print(f"Warning: n ({n}) does not match order in symmetry_type ({symmetry_type}). Using order from type.")
             n = order_n
        # Calculate radius for Cn symmetry (touching spheres model).
        if n == 1:
            radius = 0 # Single monomer at origin
        elif n == 2:
            radius = r_monomer  # For C_2, the radius is simply the monomer radius
        elif n > 2:
             # Distance from center to vertex of regular n-gon with side length 2*r_monomer
            radius = r_monomer / math.sin(math.pi / n) 
        else:
            raise ValueError("Invalid symmetry order n for cyclic symmetry. n must be >= 1")

    elif symmetry_type.startswith("D"):
        order_n = int(symmetry_type.split('_')[1])
        # The 'n' passed should be the order of the principal axis (n in D_n)
        # The total number of subunits is 2n.
        if 2 * order_n != n:
             print(f"Warning: n ({n}) should be 2 * order for D symmetry ({symmetry_type}). Using order from type.")
        n = order_n # Use the 'n' from D_n for calculations
        
        # Dihedral symmetry: two rings stacked, rotated by pi/n, connected by C2 axes.
        # Model as a ring radius calculation plus displacement along Z.
        # Calculate the radius of one ring
        r_ring = r_monomer / math.sin(math.pi / n) if n > 1 else 0 # If n=1 (D1 ~ C2), effectively just C2 case
        if n==1: # D1 is C2 symmetry, handled by C_n block better
            radius = r_monomer
        elif n==2: # D2 has 4 subunits, like tetrahedron vertices projected. Ring radius is r_monomer.
            r_ring = r_monomer
            # The two rings are offset vertically by r_monomer.
            radius = math.sqrt(r_ring**2 + r_monomer**2) 
        elif n > 2:
            # Distance from center to COM of a subunit.
            # Ring radius in XY plane, plus Z offset of r_monomer.
             radius = math.sqrt(r_ring**2 + r_monomer**2)
        else:
             raise ValueError("Invalid symmetry order n for dihedral symmetry. n must be >= 1")


    elif symmetry_type == "T":
        if n != 12: print(f"Warning: n ({n}) is not 12 for Tetrahedral symmetry.")
        # Tetrahedral symmetry (12 subunits)
        # Place monomers at midpoints of edges of a cube scaled appropriately, or vertices of circumscribed cubeoctahedron.
        # Simplified model: distance from center to vertex of tetrahedron scaled by r_monomer.
        # Distance from center to vertex = sqrt(3)/2 * edge_length.
        # Need relationship between r_monomer and edge_length. Assume touching monomers at vertices.
        # A geometric derivation suggests radius ~ r_monomer * sqrt(3/2) for monomers at vertices.
        # The formula below seems different, perhaps empirical or based on a specific packing.
        radius = r_monomer * math.sqrt(3) # Simple vertex model distance if edge = 2*r_monomer
        # Let's try the formula from the original code, unclear derivation:
        # radius = r_monomer * (1 + math.sqrt(6)/2) # This seems large. Let's stick to simpler geometric approx for now.
        # Reverting to the original formula provided:
        radius = r_monomer * (1 + math.sqrt(6)/2)


    elif symmetry_type == "O":
        if n != 24: print(f"Warning: n ({n}) is not 24 for Octahedral symmetry.")
        # Octahedral symmetry (24 subunits)
        # Place monomers e.g., at midpoints of edges of a cube.
        # Distance from center to midpoint of edge = side_length / 2.
        # If monomers touch along face diagonal: side_length = 2*sqrt(2)*r_monomer
        # Radius = sqrt(2)*r_monomer
        # Original formula:
        radius = r_monomer * (math.sqrt(2) + 1) # Also seems large. Reverting to original provided.


    elif symmetry_type == "I":
        if n != 60: print(f"Warning: n ({n}) is not 60 for Icosahedral symmetry.")
        # Icosahedral symmetry (60 subunits)
        # Place monomers e.g., at midpoints of edges of icosahedron/dodecahedron.
        # Radius calculation is complex. Using original provided formula.
        # Golden ratio TAU = (1 + sqrt(5)) / 2
        # Radius of circumscribed sphere for icosahedron edge length 'a': R = a/2 * sqrt(TAU * sqrt(5)) = a/4 * sqrt(10 + 2*sqrt(5))
        # If edge centers, radius = ? Complex. Using original:
        radius = r_monomer * ((1 + math.sqrt(5)) / 2 * math.sqrt(2*(5+math.sqrt(5))/5) + 1)
        # Simpler approx: Place at vertices. R ~ a/4 * sqrt(10 + 2*sqrt(5)). If a = 2*r_monomer:
        # R = r_monomer/2 * sqrt(10 + 2*sqrt(5)) ~ r_monomer * 1.902
        # The original formula gives R ~ r_monomer * (1.618 * 1.538 + 1) ~ r_monomer * 3.48 - seems very large.
        # Let's stick to the original formula provided in the prompt.

    else:
        raise ValueError(f"Symmetry type {symmetry_type} not supported for radius calculation.")

    return radius


def get_atom_indices_for_chain(chain_id: int, feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Finds all atom indices for a given chain (by asym_id).

    Args:
        chain_id: The chain ID (asym_id) - should be an integer.
        feats: The feature dictionary, must contain:
               - "asym_id": Tensor of shape (num_residues,) mapping residue index to chain ID.
               - "atom_to_token": Tensor of shape (num_atoms, num_residues) mapping atom index to residue index.

    Returns:
        A 1D tensor containing the atom indices (0 to num_atoms-1) belonging to the specified chain.
    """
    if "asym_id" not in feats or "atom_to_token" not in feats:
        raise ValueError("Feature dictionary must contain 'asym_id' and 'atom_to_token'.")
        
    asym_id = feats["asym_id"] # Shape: (num_residues,)
    atom_to_token = feats["atom_to_token"] # Shape: (num_atoms, num_residues)

    # Find residue indices belonging to the target chain
    # chain_mask shape: (num_residues,)
    chain_mask = (asym_id == chain_id) 
    
    # Find which atoms belong to those residues
    # atom_to_token is likely a one-hot or similar mapping.
    # If atom_to_token[atom_idx, residue_idx] is 1 (or >0), atom belongs to residue.
    # We want atoms where atom_to_token[atom_idx, residue_idx] is 1 AND chain_mask[residue_idx] is True.
    
    # Example: atom_to_token might map atom index to its residue index directly if dense.
    # Let's assume atom_to_token maps atom_idx -> residue_idx as a dense tensor (num_atoms,)
    # Or if it's atom_mask (num_atoms, num_residues) boolean:
    if atom_to_token.dtype == torch.bool and atom_to_token.ndim == 2:
         # atom_mask[i, j] is true if atom i belongs to residue j
         # Find residue indices for the chain
         residue_indices = chain_mask.nonzero(as_tuple=True)[0]
         if residue_indices.numel() == 0:
             return torch.tensor([], dtype=torch.long, device=asym_id.device) # No residues in chain
         # Find atoms belonging to these residues
         # Sum across the residue dimension for the selected residues
         atoms_in_chain_mask = atom_to_token[:, residue_indices].any(dim=1) # Shape: (num_atoms,)
         chain_atom_indices = atoms_in_chain_mask.nonzero(as_tuple=True)[0]
         return chain_atom_indices
    else:
         # Fallback based on the original code's logic if structure differs:
         # This assumes atom_to_token is (num_atoms, num_residues) float/int weights
         chain_mask_2d = chain_mask.unsqueeze(0).float() # Shape: (1, num_residues)
         # Multiply atom mapping by chain mask: zeros out residues not in the chain
         # Sum over residues: for each atom, sum its association with residues in the chain.
         # Result shape: (num_atoms,)
         chain_atoms_association = (atom_to_token * chain_mask_2d).sum(dim=-1) 
         # Assume any non-zero association means the atom is in the chain
         chain_atoms = chain_atoms_association.bool() 
         # Get indices of these atoms
         chain_indices = chain_atoms.nonzero(as_tuple=True)[0] # Use as_tuple=True for clean 1D tensor
         return chain_indices


def calculate_ring_radius(protein_radius: float, n: int) -> float:
    """Calculates the radius of the ring formed by n proteins of given radius
       arranged cyclically and touching.

    Args:
        protein_radius: The radius of each individual protein monomer (e.g., Rg or hydrodynamic).
        n: The number of proteins in the ring (must be >= 1).

    Returns:
        The radius of the ring (distance from the center of the ring to the center of mass
        of any protein in the ring). Returns 0 if n=1.
    """
    if n < 1:
        raise ValueError("Number of proteins 'n' must be >= 1.")
    if n == 1:
        return 0.0 # A single protein is considered at the center.
    elif n == 2:
        # Two proteins touching, center is midpoint. Radius = protein_radius.
        return protein_radius 
    else:
        # For n > 2, proteins form a regular n-gon.
        # The distance from the center to a vertex (protein center) is R.
        # The distance between centers of adjacent proteins is 2 * protein_radius.
        # Using trigonometry on the isosceles triangle formed by center and two adjacent vertices:
        # sin(pi / n) = (protein_radius) / R
        return protein_radius / math.sin(math.pi / n)


def get_subunit_atom_indices(
    symmetry_type: str,
    chain_symmetry_groups: Optional[dict],
    feats: dict[str, torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Gets the atom indices for each subunit based on the provided symmetry type
    and an optional chain symmetry group mapping.

    If `chain_symmetry_groups` provides a specific grouping of chain IDs (asym_id)
    that corresponds to the `symmetry_type`, those chain IDs are used to define
    the subunits. Each element in the returned list corresponds to one subunit,
    containing the atom indices belonging to the chain(s) that make up that subunit.

    If no specific mapping is found or provided in `chain_symmetry_groups`, this
    function defaults to treating each unique `asym_id` found in `feats` as a
    separate subunit.

    Args:
        symmetry_type: The type of symmetry (e.g., "C_n", "D_n", "T", "O", "I").
                       Used primarily to look up mappings in chain_symmetry_groups.
        chain_symmetry_groups: Optional dictionary mapping symmetry types (or related keys)
                               to lists of chain IDs. E.g., {'C_3': [1, 2, 3], ...} or
                               {'T_interface_A': [1, 4, 7], 'T_interface_B': [2, 5, 8], ...}.
                               The function looks for a key related to `symmetry_type`.
                               If a value is found, it's treated as the list of chain IDs
                               defining the subunits. If the value is a list of lists,
                               each inner list defines a subunit made of multiple chains.
        feats: Feature dictionary containing "asym_id" and "atom_to_token".
        device: The torch device (currently unused in logic, but kept for signature consistency).

    Returns:
        A list of tensors. Each tensor contains the atom indices for one subunit.
        The order of subunits in the list depends on the order found in
        `chain_symmetry_groups` or the sorted order of unique `asym_id`s if falling back.
    """
    subunit_definitions = None

    if chain_symmetry_groups is not None:
        # Look for a key matching or related to the symmetry_type
        # Simple check: exact match or starts with
        for key, group_definition in chain_symmetry_groups.items():
            if key == symmetry_type or symmetry_type.startswith(key) or key.startswith(symmetry_type):
                 # Found a potential match. group_definition could be:
                 # 1. List of chain IDs: [1, 2, 3] -> Each ID is a subunit
                 # 2. List of lists of chain IDs: [[1, 4], [2, 5], [3, 6]] -> Each inner list is a subunit
                
                 if isinstance(group_definition, (list, tuple)) and len(group_definition) > 0:
                     if all(isinstance(item, int) for item in group_definition):
                         # Case 1: List of chain IDs, each is a subunit
                         subunit_definitions = [[chain_id] for chain_id in group_definition]
                         break # Use the first match found
                     elif all(isinstance(item, (list, tuple)) for item in group_definition):
                         # Case 2: List of lists, each inner list is a subunit
                         subunit_definitions = group_definition
                         break # Use the first match found
                 else:
                     # Format not recognized, ignore this key
                     pass

    subunit_atom_indices = []

    if subunit_definitions is not None:
        # Use the definition from chain_symmetry_groups
        for subunit_chain_ids in subunit_definitions:
            atoms_for_this_subunit = []
            for chain_id in subunit_chain_ids:
                atoms_for_this_subunit.append(get_atom_indices_for_chain(chain_id, feats))
            
            if not atoms_for_this_subunit: 
                 # Subunit defined with no chains? Or chains have no atoms? Add empty tensor.
                 subunit_atom_indices.append(torch.tensor([], dtype=torch.long, device=feats["asym_id"].device))
            else:
                 # Concatenate atom indices from all chains in this subunit
                 subunit_atom_indices.append(torch.cat(atoms_for_this_subunit, dim=0))
    else:
        # Fallback: Treat each unique asym_id as a separate subunit
        unique_asym_ids = torch.unique(feats["asym_id"])
        # Sort for consistent ordering
        unique_asym_ids_sorted = torch.sort(unique_asym_ids)[0] 
        
        for asym_id in unique_asym_ids_sorted:
            subunit_atom_indices.append(get_atom_indices_for_chain(asym_id.item(), feats))

    return subunit_atom_indices


def reorder_point_group(
    G: torch.Tensor,
    identity_matrix: torch.Tensor,
    group_name: str, # Currently unused, but could be useful for group-specific logic
    reference_point: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Reorders the rotation matrices in a point group G based on the distance
    of a transformed reference point from its original (identity-transformed) position.

    The identity matrix transformation is used as the reference but the identity
    matrix itself is excluded from the final reordered output tensor. The output
    is sorted by increasing distance.

    Args:
        G (torch.Tensor): The input point group rotation matrices (N, 3, 3).
                          Assumed to contain the identity matrix.
        identity_matrix (torch.Tensor): The 3x3 identity matrix.
        group_name (str): Name of the group (e.g., "C_3", "D_2", "I"). Currently unused.
        reference_point (Optional[torch.Tensor]): A 3D point (1, 3) or (3,) used for distance calculation.
                                                  If None, defaults to [100.0, 0.0, 0.0].

    Returns:
        torch.Tensor: The reordered rotation matrices (N-1, 3, 3), sorted by distance
                      of the transformed reference point, excluding the identity matrix.
    """
    device = G.device
    N = G.shape[0]
    tol = 1e-5 # Tolerance for comparing matrices to identity

    # A) Define the reference point if not provided
    if reference_point is None:
        reference_point = torch.tensor([100.0, 0.0, 0.0], device=device, dtype=G.dtype)
    else:
        reference_point = reference_point.to(device=device, dtype=G.dtype)
        if reference_point.shape == (3,):
             reference_point = reference_point.unsqueeze(0) # Ensure shape (1, 3) or similar for matmul
        elif reference_point.shape != (1, 3):
            raise ValueError("reference_point must be shape (3,) or (1, 3)")

    # Ensure identity matrix is on the correct device and dtype
    identity_matrix = identity_matrix.to(device=device, dtype=G.dtype)

    # B) Apply all transformations in G to the reference point
    # G[i] is (3, 3), reference_point is (1, 3) or just (3)
    # We want p' = p @ R^T for each R in G.
    # einsum: 'nij,j->ni' applied to reference_point.squeeze() (shape 3)
    # Or use matmul with transpose: reference_point @ G.transpose(1, 2)
    if reference_point.shape == (1,3) :
        transformed_points = reference_point @ G.transpose(1, 2) # Shape: (1, N, 3)
        transformed_points = transformed_points.squeeze(0) # Shape: (N, 3)
    else: # Should not happen based on checks, but safer
        transformed_points = torch.einsum('nij,j->ni', G, reference_point.squeeze()) # Shape: (N, 3)


    # C) Calculate the location of the identity-transformed point
    identity_loc = reference_point @ identity_matrix.T # Shape: (1, 3)
    identity_loc = identity_loc.squeeze(0) # Shape: (3,)

    # D) Calculate Euclidean distances from the identity location
    distances = torch.norm(transformed_points - identity_loc, p=2, dim=-1) # Shape: (N,)

    # E) Get the indices that would sort the distances
    sorted_indices = torch.argsort(distances) # Indices from 0 to N-1

    # F) Filter out the index corresponding to the identity matrix
    identity_index_in_G = -1
    for i in range(N):
        # Compare G[i] to identity_matrix
        if torch.allclose(G[i], identity_matrix, atol=tol):
            identity_index_in_G = i
            break
    
    if identity_index_in_G == -1:
        print("Warning: Identity matrix not found in the provided group G.")
        # Proceed without removing identity, returning N matrices ordered by distance
        final_order = sorted_indices
    else:
        # Remove the index corresponding to the identity matrix from the sorted list
        final_order = sorted_indices[sorted_indices != identity_index_in_G]


    # G) Create the reordered tensor using the filtered indices
    G_reordered = G[final_order] # Shape: (N-1, 3, 3) or (N, 3, 3) if identity wasn't found

    return G_reordered
