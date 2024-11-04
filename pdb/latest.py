import os
import gzip
from Bio import PDB
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

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
        
        # Track structure statistics
        self.structure_stats = {
            'total_files': 0,
            'files_with_2_chains': 0,
            'files_with_more_chains': 0,
            'files_with_x_chains': defaultdict(int),  # New: Track each specific chain count
            'chain_counts': defaultdict(int),
            'processed_files': [],
            'failed_files': []
        }

    # [Previous methods remain unchanged: interpolate_coordinates, has_valid_coordinates, get_residue_coords]
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
            avg_distance = 3.8 if residue_list[missing_idx].get_resname() in self.protein_residues else 6.0
            direction = np.array([1.0, 0.0, 0.0])
            return prev_coords + direction * avg_distance, 'extended_from_prev'
            
        # If we only found next valid neighbor
        elif next_idx < len(residue_list):
            next_coords = self.get_residue_coords(residue_list[next_idx])
            avg_distance = 3.8 if residue_list[missing_idx].get_resname() in self.protein_residues else 6.0
            direction = np.array([-1.0, 0.0, 0.0])
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

    def extract_data(self, pdb_file, is_gzipped=False):
        """Extract sequences and coordinates from PDB file with missing coordinate handling."""
        # Create a temporary file if gzipped
        if is_gzipped:
            with gzip.open(pdb_file, 'rt') as f:
                content = f.read()
            temp_pdb = os.path.join(os.path.dirname(pdb_file), 'temp.pdb')
            with open(temp_pdb, 'w') as f:
                f.write(content)
            structure = self.parser.get_structure('structure', temp_pdb)
            os.remove(temp_pdb)
        else:
            structure = self.parser.get_structure('structure', pdb_file)
            
        chain_data = defaultdict(list)
        sequences = defaultdict(str)
        
        model = structure[0]
        num_chains = len(list(model.get_chains()))
        
        # Update chain statistics
        if num_chains == 2:
            self.structure_stats['files_with_2_chains'] += 1
        elif num_chains > 2:
            self.structure_stats['files_with_more_chains'] += 1
        
        # New: Track specific chain counts
        self.structure_stats['files_with_x_chains'][num_chains] += 1
        self.structure_stats['chain_counts'][num_chains] += 1
        
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
                
                if not self.has_valid_coordinates(residue):
                    self.missing_coords_stats[f"{chain_type}_{res_name}"] += 1
                    coords, coord_status = self.interpolate_coordinates(residue_list, i)
                    if coords is None:
                        continue
                else:
                    coords = self.get_residue_coords(residue)
                    coord_status = 'observed'
                
                entry = {
                    'pdb_id': os.path.basename(pdb_file).split('.')[0].split('_')[0],
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

    def process_directory(self, directory_path, output_dir='final_output'):
        """Process all PDB and PDB.gz files in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        all_data = []
        sequences_dict = {}
        
        # Find all PDB and PDB.gz files
        pdb_files = list(Path(directory_path).glob('*.pdb')) + list(Path(directory_path).glob('*.pdb.gz'))
        self.structure_stats['total_files'] = len(pdb_files)
        
        print(f"Found {len(pdb_files)} PDB files to process...")
        
        # Create progress bar
        progress_bar = tqdm(total=len(pdb_files), desc="Processing PDB files", 
                          unit="file", position=0, leave=True)
        
        for pdb_file in pdb_files:
            pdb_file = str(pdb_file)
            
            try:
                is_gzipped = pdb_file.endswith('.gz')
                chain_data, sequences = self.extract_data(pdb_file, is_gzipped)
                
                for chain_id, data in chain_data.items():
                    all_data.extend(data)
                sequences_dict[os.path.basename(pdb_file).split('.')[0].split('_')[0]] = sequences
                
                self.structure_stats['processed_files'].append(pdb_file)
                
            except Exception as e:
                print(f"\nError processing {pdb_file}: {str(e)}")
                self.structure_stats['failed_files'].append(pdb_file)
            
            # Update progress bar
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()

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
            
            # Save structure statistics
            struct_stats_path = os.path.join(output_dir, 'structure_statistics.txt')
            with open(struct_stats_path, 'w') as f:
                f.write(f"Total files processed: {self.structure_stats['total_files']}\n")
                f.write(f"Files with exactly 2 chains: {self.structure_stats['files_with_2_chains']}\n")
                f.write(f"Files with more than 2 chains: {self.structure_stats['files_with_more_chains']}\n")
                
                # New: Write detailed chain count statistics
                f.write("\nDetailed chain count distribution:\n")
                for chain_count, num_files in sorted(self.structure_stats['files_with_x_chains'].items()):
                    f.write(f"Files with {chain_count} chains: {num_files}\n")
                
                f.write(f"\nSuccessfully processed files: {len(self.structure_stats['processed_files'])}\n")
                f.write(f"Failed files: {len(self.structure_stats['failed_files'])}\n")
                if self.structure_stats['failed_files']:
                    f.write("\nFailed files list:\n")
                    for failed_file in self.structure_stats['failed_files']:
                        f.write(f"- {failed_file}\n")
            
            # Save coordinates
            parquet_path = os.path.join(output_dir, 'coordinates.parquet')
            df.to_parquet(parquet_path, index=False)
            
            csv_path = os.path.join(output_dir, 'coordinates.csv')
            df.to_csv(csv_path, index=False)
            
            # Save sequences
            sequences_path = os.path.join(output_dir, 'sequences.txt')
            with open(sequences_path, 'w') as f:
                for pdb_id, chains in sequences_dict.items():
                    f.write(f">{pdb_id}\n")
                    for chain_id, sequence in chains.items():
                        f.write(f"Chain {chain_id}: {sequence}\n")
            
            print("\nProcessing complete! Summary:")
            print(f"Total files found: {self.structure_stats['total_files']}")
            print(f"Files with 2 chains: {self.structure_stats['files_with_2_chains']}")
            print(f"Files with >2 chains: {self.structure_stats['files_with_more_chains']}")
            
            # New: Print detailed chain count statistics
            print("\nDetailed chain count distribution:")
            for chain_count, num_files in sorted(self.structure_stats['files_with_x_chains'].items()):
                print(f"Files with {chain_count} chains: {num_files}")
            
            print(f"\nSuccessfully processed: {len(self.structure_stats['processed_files'])}")
            print(f"Failed to process: {len(self.structure_stats['failed_files'])}")
            
            return df
        
        return None

if __name__ == "__main__":
    processor = MLStructureProcessor()
    dataset = processor.process_directory("/blue/xiaofan/gangavarapus/pdb_data/data")