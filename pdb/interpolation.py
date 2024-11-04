import os
from Bio import PDB
import pandas as pd
import numpy as np
from collections import defaultdict

class MLStructureProcessor:
    def __init__(self):
        self.parser = PDB.PDBParser(QUIET=True)
        self.protein_residues = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                               'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                               'THR', 'TRP', 'TYR', 'VAL'}
        self.rna_residues = {'A', 'C', 'G', 'U'}
        
        self.protein_residue_encoding = {res: idx for idx, res in enumerate(sorted(self.protein_residues))}
        self.rna_residue_encoding = {res: idx for idx, res in enumerate(sorted(self.rna_residues))}
        
        # Track missing coordinate statistics
        self.missing_coords_stats = defaultdict(int)

    def interpolate_coordinates(self, residue_list, missing_idx):
        """Interpolate coordinates using neighboring residues."""
        prev_idx = missing_idx - 1
        next_idx = missing_idx + 1
        
        # Find valid previous coordinates
        while prev_idx >= 0:
            prev_residue = residue_list[prev_idx]
            if self.has_valid_coordinates(prev_residue):
                break
            prev_idx -= 1
            
        # Find valid next coordinates
        while next_idx < len(residue_list):
            next_residue = residue_list[next_idx]
            if self.has_valid_coordinates(next_residue):
                break
            next_idx += 1
            
        # If we found both valid neighbors
        if prev_idx >= 0 and next_idx < len(residue_list):
            prev_coords = self.get_residue_coords(residue_list[prev_idx])
            next_coords = self.get_residue_coords(residue_list[next_idx])
            
            # Linear interpolation
            weight = (missing_idx - prev_idx) / (next_idx - prev_idx)
            interpolated = prev_coords + weight * (next_coords - prev_coords)
            return interpolated, 'interpolated'
            
        # If we only found previous valid neighbor
        elif prev_idx >= 0:
            prev_coords = self.get_residue_coords(residue_list[prev_idx])
            # Use average CA-CA or C1'-C1' distance (~ 3.8Å for proteins, ~ 6.0Å for RNA)
            avg_distance = 3.8 if residue_list[missing_idx].get_resname() in self.protein_residues else 6.0
            direction = np.array([1.0, 0.0, 0.0])  # Arbitrary direction
            return prev_coords + direction * avg_distance, 'extended_from_prev'
            
        # If we only found next valid neighbor
        elif next_idx < len(residue_list):
            next_coords = self.get_residue_coords(residue_list[next_idx])
            avg_distance = 3.8 if residue_list[missing_idx].get_resname() in self.protein_residues else 6.0
            direction = np.array([-1.0, 0.0, 0.0])  # Arbitrary direction
            return next_coords + direction * avg_distance, 'extended_from_next'
            
        return None, 'completely_missing'

    def has_valid_coordinates(self, residue):
        """Check if residue has valid coordinates for its reference atom."""
        try:
            atom_name = 'CA' if residue.get_resname() in self.protein_residues else "C1'"
            residue[atom_name].get_coord()
            return True
        except (KeyError, AttributeError):
            return False

    def get_residue_coords(self, residue):
        """Get coordinates for the reference atom of a residue."""
        atom_name = 'CA' if residue.get_resname() in self.protein_residues else "C1'"
        return residue[atom_name].get_coord()

    def extract_data(self, pdb_file):
        """Extract sequences and coordinates from PDB file with missing coordinate handling."""
        structure = self.parser.get_structure('structure', pdb_file)
        chain_data = defaultdict(list)
        sequences = defaultdict(str)
        
        model = structure[0]
        
        for chain in model:
            chain_id = chain.id
            residue_list = list(chain)
            
            for i, residue in enumerate(residue_list):
                res_name = residue.get_resname()
                
                if res_name in self.protein_residues:
                    chain_type = 'protein'
                    atom_name = 'CA'
                    residue_encoding = self.protein_residue_encoding[res_name]
                elif res_name in self.rna_residues:
                    chain_type = 'RNA'
                    atom_name = "C1'"
                    residue_encoding = self.rna_residue_encoding[res_name]
                else:
                    continue
                
                sequences[chain_id] += res_name
                
                # Handle missing coordinates
                if not self.has_valid_coordinates(residue):
                    self.missing_coords_stats[f"{chain_type}_{res_name}"] += 1
                    coords, coord_status = self.interpolate_coordinates(residue_list, i)
                    if coords is None:
                        continue
                else:
                    coords = self.get_residue_coords(residue)
                    coord_status = 'observed'
                
                entry = {
                    'pdb_id': os.path.basename(pdb_file).split('.')[0],
                    'chain_id': chain_id,
                    'chain_type': chain_type,
                    'residue': res_name,
                    'residue_encoded': residue_encoding,
                    'position': residue.id[1],
                    'relative_position': i / len(residue_list),
                    'x': coords[0],
                    'y': coords[1],
                    'z': coords[2],
                    'coordinate_status': coord_status
                }
                
                chain_data[chain_id].append(entry)
        
        return chain_data, sequences

    def process_pdb_files(self, pdb_files, output_dir='interpolation_for_missing_data'):
        """Process multiple PDB files with missing coordinate handling."""
        os.makedirs(output_dir, exist_ok=True)
        all_data = []
        sequences_dict = {}
        
        for pdb_file in pdb_files:
            if not os.path.exists(pdb_file):
                print(f"File not found: {pdb_file}")
                continue
                
            print(f"Processing {pdb_file}...")
            
            try:
                chain_data, sequences = self.extract_data(pdb_file)
                
                for chain_id, data in chain_data.items():
                    all_data.extend(data)
                sequences_dict[os.path.basename(pdb_file).split('.')[0]] = sequences
                
            except Exception as e:
                print(f"Error processing {pdb_file}: {str(e)}")
                continue

        if all_data:
            # Create DataFrame
            df = pd.DataFrame(all_data)
            
            # Save missing coordinate statistics
            stats_path = os.path.join(output_dir, 'missing_coordinates_stats.csv')
            stats_df = pd.DataFrame([
                {'residue_type': k, 'count': v} 
                for k, v in self.missing_coords_stats.items()
            ])
            stats_df.to_csv(stats_path, index=False)
            print(f"Missing coordinate statistics saved to {stats_path}")
            
            # Save coordinates in both formats
            parquet_path = os.path.join(output_dir, 'coordinates.parquet')
            df.to_parquet(parquet_path, index=False)
            print(f"Coordinates saved to {parquet_path}")
            
            csv_path = os.path.join(output_dir, 'coordinates.csv')
            df.to_csv(csv_path, index=False)
            print(f"Coordinates also saved as CSV to {csv_path}")
            
            # Save sequences
            sequences_path = os.path.join(output_dir, 'sequences.txt')
            with open(sequences_path, 'w') as f:
                for pdb_id, chains in sequences_dict.items():
                    f.write(f">{pdb_id}\n")
                    for chain_id, sequence in chains.items():
                        f.write(f"Chain {chain_id}: {sequence}\n")
            print(f"Sequences saved to {sequences_path}")
            
            return df
        
        return None
    
if __name__ == "__main__":
	processor = MLStructureProcessor()
	pdb_files = ['5www.pdb', '8rxd.pdb']  # We can add our PDB file paths here
	dataset = processor.process_pdb_files(pdb_files)