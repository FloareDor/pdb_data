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
        
        # Create residue encodings for ML
        self.protein_residue_encoding = {res: idx for idx, res in enumerate(sorted(self.protein_residues))}
        self.rna_residue_encoding = {res: idx for idx, res in enumerate(sorted(self.rna_residues))}

    def extract_data(self, pdb_file):
        """Extract sequences and coordinates from PDB file with ML-friendly features."""
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
                
                try:
                    atom = residue[atom_name]
                    coords = atom.get_coord()
                    sequences[chain_id] += res_name
                    
                    # Calculate additional ML features
                    prev_coords = None
                    next_coords = None
                    
                    if i > 0 and residue_list[i-1].get_resname() in (self.protein_residues | self.rna_residues):
                        try:
                            prev_coords = residue_list[i-1][atom_name].get_coord()
                        except KeyError:
                            pass
                            
                    if i < len(residue_list)-1 and residue_list[i+1].get_resname() in (self.protein_residues | self.rna_residues):
                        try:
                            next_coords = residue_list[i+1][atom_name].get_coord()
                        except KeyError:
                            pass

                    entry = {
                        'pdb_id': os.path.basename(pdb_file).split('.')[0],
                        'chain_id': chain_id,
                        'chain_type': chain_type,
                        'residue': res_name,
                        'residue_encoded': residue_encoding,
                        'position': residue.id[1],
                        'relative_position': i / len(residue_list),  # Normalized position
                        'x': coords[0],
                        'y': coords[1],
                        'z': coords[2],
                    }
                    
                    # Add distance features if available
                    if prev_coords is not None:
                        dist = np.linalg.norm(coords - prev_coords)
                        entry['prev_residue_distance'] = dist
                    
                    if next_coords is not None:
                        dist = np.linalg.norm(coords - next_coords)
                        entry['next_residue_distance'] = dist
                    
                    chain_data[chain_id].append(entry)
                    
                except KeyError:
                    continue
        
        return chain_data, sequences

    def process_pdb_files(self, pdb_files, output_dir='processed_data_ml'):
        """Process multiple PDB files and create ML dataset with improved features."""
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
            # Create DataFrame with optimized dtypes
            df = pd.DataFrame(all_data)
            
            # Optimize numeric columns
            numeric_columns = ['position', 'relative_position', 'x', 'y', 'z', 
                             'residue_encoded', 'prev_residue_distance', 'next_residue_distance']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Save as parquet for ML
            parquet_path = os.path.join(output_dir, 'coordinates.parquet')
            df.to_parquet(parquet_path, index=False)
            print(f"Coordinates saved to {parquet_path}")
            
            # Also save as CSV for human readability
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