import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

class NiftiMRIDParser:
    def __init__(self, modality_keywords=None, modality_synonyms=None):
        """
        Initialize NIFTI parser with configurable modality keywords and synonyms.
        
        Args:
            modality_keywords: list of modality patterns to strip from filenames
            modality_synonyms: dict mapping canonical name -> list of synonyms
                              e.g., {'FLAIR': ['FL', 'FLAIR'], 'T1': ['T1', 'T1W']}
        """
        # Default modality keywords to strip from filenames
        self.modality_keywords = modality_keywords or [
            'T1', 'T1W', 'T1C', 'T1CE', 'T1POST',
            'T2', 'T2W', 'T2FLAIR',
            'FLAIR', 'FL',
            'DWI', 'ADC', 'SWI',
            'DLMUSE', 'LPS', 'MUSE', 'SEG', 'MASK', 'BRAIN'
        ]
        
        # Default modality synonyms - maps canonical name to list of alternatives
        # All variants will be normalized to the canonical name
        self.modality_synonyms = modality_synonyms or {
            'FLAIR': ['FLAIR', 'FL', 'T2FLAIR'],
            'T1': ['T1', 'T1W'],
            'T1C': ['T1C', 'T1CE', 'T1POST', 'T1CONTRAST'],
            'T2': ['T2', 'T2W'],
            'DWI': ['DWI', 'DIFFUSION'],
            'ADC': ['ADC', 'ADCMAP'],
            'SWI': ['SWI', 'SUSCEPTIBILITY'],
        }
        
        # Build reverse mapping: synonym -> canonical name (case-insensitive)
        self._synonym_to_canonical = {}
        for canonical, synonyms in self.modality_synonyms.items():
            for syn in synonyms:
                self._synonym_to_canonical[syn.upper()] = canonical.upper()
        
        # Compile regex for efficient matching
        self.modality_pattern = re.compile(
            r'[_\-\s](' + '|'.join(self.modality_keywords) + r')(?:[_\-\s]|$)',
            re.IGNORECASE
        )
        
        self.df = None  # Store the DataFrame for querying
        self._modality_dirs = {}  # Store directory -> canonical modality mapping
        
    def _normalize_modality_name(self, modality_name):
        """
        Normalize a modality name to its canonical form using synonyms.
        Case-insensitive.
        
        Examples:
            'flair' -> 'FLAIR'
            'FL' -> 'FLAIR'
            't1w' -> 'T1'
            'T1' -> 'T1'
        """
        modality_upper = modality_name.upper()
        
        # Check if it's a known synonym
        if modality_upper in self._synonym_to_canonical:
            return self._synonym_to_canonical[modality_upper]
        
        # Check if any synonym matches
        for canonical, synonyms in self.modality_synonyms.items():
            if modality_upper in [s.upper() for s in synonyms]:
                return canonical.upper()
        
        # Return as-is if no synonym found
        return modality_upper
        
    def extract_mrid(self, filename):
        """
        Extract MRID from filename by stripping modality keywords and extensions.
        
        Handles derived files by only stripping the FIRST modality keyword and
        everything after it, preventing false MRIDs from complex suffixes.
        
        Examples:
            'SubjectA_T1.nii.gz' -> 'SubjectA'
            'Subject-A_FLAIR_LPS.nii' -> 'Subject-A'
            'Subject001_T1_LPS_DLMUSE.nii.gz' -> 'Subject001'
            '001_T1_T1_DLMUSE.nii.gz' -> '001'
        """
        # Remove .nii.gz or .nii extension
        stem = filename.replace('.nii.gz', '').replace('.nii', '')
        
        # Find the FIRST occurrence of any modality keyword
        # This handles derived files like Subject001_T1_LPS_DLMUSE.nii.gz
        match = self.modality_pattern.search(stem)
        
        if match:
            # Extract everything BEFORE the first modality keyword
            mrid = stem[:match.start()]
        else:
            # No modality keywords found, use the whole stem
            mrid = stem
        
        # Clean up trailing/leading separators
        mrid = mrid.strip('_- ')
        
        return mrid if mrid else filename  # Fallback to original if empty
    
    def scan_directory(self, directory_path, modality_name):
        """
        Scan a directory and return mapping of MRID -> file info.
        
        Args:
            directory_path: path to directory containing NIFTI files
            modality_name: the canonical modality name for this directory
        """
        directory = Path(directory_path)
        mrid_map = {}
        
        # Normalize the modality name using synonyms
        canonical_modality = self._normalize_modality_name(modality_name)
        
        for filepath in directory.glob('*.nii*'):
            if filepath.is_file():
                filename = filepath.name
                mrid = self.extract_mrid(filename)
                
                if mrid in mrid_map:
                    # Collision detection - warn user
                    print(f"WARNING: Multiple files map to MRID '{mrid}' in {modality_name} ({canonical_modality}):")
                    print(f"  - {mrid_map[mrid]['filename']}")
                    print(f"  - {filename}")
                    print(f"  Using the first file. If this is incorrect, rename files to be unique.")
                else:
                    mrid_map[mrid] = {
                        'filename': filename,
                        'filepath': str(filepath),
                        'modality': canonical_modality  # Use canonical name
                    }
        
        return mrid_map
    
    def create_master_csv(self, directory_configs, output_csv='master_index.csv'):
        """
        Create master CSV linking MRIDs across directories.
        
        Args:
            directory_configs: dict like {'T1': '/path/to/t1', 'FLAIR': '/path/to/flair'}
                              Keys can be any modality name (including synonyms)
        
        Returns:
            DataFrame with MRID and file information
        """
        # Collect all MRIDs and their files
        all_data = defaultdict(dict)
        
        # Store the directory -> canonical modality mapping
        self._modality_dirs = {}
        
        for modality_name, directory_path in directory_configs.items():
            # Normalize modality name
            canonical_modality = self._normalize_modality_name(modality_name)
            self._modality_dirs[directory_path] = canonical_modality
            
            print(f"\nScanning {modality_name} -> {canonical_modality}: {directory_path}")
            mrid_map = self.scan_directory(directory_path, modality_name)
            
            for mrid, file_info in mrid_map.items():
                # Use canonical modality name in the data structure
                canonical_mod = file_info['modality']
                
                # Handle case where same MRID+modality appears in multiple dirs
                # (e.g., both FL/ and FLAIR/ directories with overlapping files)
                if canonical_mod in all_data[mrid]:
                    print(f"WARNING: MRID '{mrid}' already has {canonical_mod} data:")
                    print(f"  Existing: {all_data[mrid][canonical_mod]}")
                    print(f"  New: {file_info['filename']} (from {modality_name} directory)")
                    print(f"  Keeping existing file.")
                else:
                    all_data[mrid][canonical_mod] = file_info['filename']
                    all_data[mrid][f'{canonical_mod}_path'] = file_info['filepath']
        
        # Convert to DataFrame
        rows = []
        for mrid in sorted(all_data.keys()):
            row = {'MRID': mrid}
            row.update(all_data[mrid])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Reorder columns: MRID first, then alphabetically
        cols = ['MRID'] + sorted([c for c in df.columns if c != 'MRID'])
        df = df[cols]
        
        # Store DataFrame for querying
        self.df = df
        
        df.to_csv(output_csv, index=False)
        print(f"\n{'='*60}")
        print(f"Created master CSV: {output_csv}")
        print(f"Total MRIDs found: {len(df)}")
        
        # Report missing data
        modality_cols = [col for col in df.columns if col != 'MRID' and not col.endswith('_path')]
        if modality_cols:
            print(f"\nModality coverage:")
            for col in modality_cols:
                present = df[col].notna().sum()
                missing = df[col].isna().sum()
                pct = (present / len(df) * 100) if len(df) > 0 else 0
                print(f"  {col:15s}: {present:3d} present, {missing:3d} missing ({pct:5.1f}%)")
        print(f"{'='*60}")
        
        return df
    
    def load_csv(self, csv_path):
        """Load an existing master CSV for querying."""
        self.df = pd.read_csv(csv_path)
        return self.df
    
    def _normalize_modality(self, modality):
        """
        Normalize modality name to match column names (case-insensitive + synonyms).
        
        Args:
            modality: modality name (can be synonym)
            
        Returns:
            Canonical column name from DataFrame
            
        Raises:
            ValueError if modality not found
        """
        if self.df is None:
            raise ValueError("No data loaded. Run create_master_csv() or load_csv() first.")
        
        # First normalize using synonyms
        canonical = self._normalize_modality_name(modality)
        
        # Get all non-path modality columns
        available = [col for col in self.df.columns if col != 'MRID' and not col.endswith('_path')]
        
        # Case-insensitive match against DataFrame columns
        canonical_upper = canonical.upper()
        for col in available:
            if col.upper() == canonical_upper:
                return col
        
        raise ValueError(f"Modality '{modality}' (normalized to '{canonical}') not found. Available: {available}")
    
    def filter_by_modalities(self, required_modalities, return_df=False):
        """
        Filter subjects that have ALL required modalities.
        
        Args:
            required_modalities: list of modality names (case-insensitive, synonyms OK)
            return_df: if True, return filtered DataFrame; if False, return MRIDs list
            
        Returns:
            dict with keys 'mrids', 'count', 'dataframe' (if requested)
        """
        if self.df is None:
            raise ValueError("No data loaded. Run create_master_csv() or load_csv() first.")
        
        # Normalize modality names (handles case + synonyms)
        normalized = [self._normalize_modality(mod) for mod in required_modalities]
        
        # Filter rows where all required modalities are non-null
        mask = pd.Series([True] * len(self.df))
        for modality in normalized:
            mask &= self.df[modality].notna()
        
        filtered_df = self.df[mask]
        mrids = filtered_df['MRID'].tolist()
        
        result = {
            'mrids': mrids,
            'count': len(mrids)
        }
        
        if return_df:
            result['dataframe'] = filtered_df
        
        return result
    
    def get_modality_coverage(self):
        """
        Get coverage statistics for each modality.
        
        Returns:
            dict mapping modality -> {'count': int, 'percentage': float, 'mrids': list}
        """
        if self.df is None:
            raise ValueError("No data loaded. Run create_master_csv() or load_csv() first.")
        
        total = len(self.df)
        coverage = {}
        
        for col in self.df.columns:
            if col != 'MRID' and not col.endswith('_path'):
                count = self.df[col].notna().sum()
                mrids = self.df[self.df[col].notna()]['MRID'].tolist()
                coverage[col] = {
                    'count': count,
                    'percentage': (count / total * 100) if total > 0 else 0,
                    'mrids': mrids
                }
        
        return coverage
    
    def compare_modality_sets(self, *modality_sets):
        """
        Compare counts across different modality requirement sets.
        
        Args:
            *modality_sets: variable number of modality lists (synonyms OK)
            
        Returns:
            list of dicts with 'modalities', 'count', 'mrids' for each set
        """
        results = []
        
        for modality_set in modality_sets:
            filtered = self.filter_by_modalities(modality_set)
            # Normalize for display
            normalized_set = [self._normalize_modality(m) for m in modality_set]
            results.append({
                'modalities': normalized_set,
                'count': filtered['count'],
                'mrids': filtered['mrids']
            })
        
        return results
    
    def get_path(self, mrid, modality):
        """
        Get the file path for a specific MRID and modality.
        
        Args:
            mrid: the subject ID
            modality: the modality name (case-insensitive, synonyms OK)
            
        Returns:
            str: file path, or None if not found
        """
        if self.df is None:
            raise ValueError("No data loaded. Run create_master_csv() or load_csv() first.")
        
        # Normalize modality name (handles case + synonyms)
        normalized = self._normalize_modality(modality)
        path_col = f"{normalized}_path"
        
        # Find the row for this MRID
        row = self.df[self.df['MRID'] == mrid]
        
        if row.empty:
            raise ValueError(f"MRID '{mrid}' not found in dataset")
        
        # Check if path column exists
        if path_col not in self.df.columns:
            # Fall back to just returning the filename if no path column
            if normalized in self.df.columns:
                filename = row[normalized].iloc[0]
                return filename if pd.notna(filename) else None
            raise ValueError(f"No path information for modality '{modality}'")
        
        path = row[path_col].iloc[0]
        
        if pd.isna(path):
            return None
        
        return path
    
    def get_paths(self, mrid, modalities=None):
        """
        Get paths for multiple modalities for a given MRID.
        
        Args:
            mrid: the subject ID
            modalities: list of modality names (synonyms OK), or None for all available
            
        Returns:
            dict mapping canonical modality -> path (None if missing)
        """
        if self.df is None:
            raise ValueError("No data loaded. Run create_master_csv() or load_csv() first.")
        
        if modalities is None:
            # Get all modalities
            modalities = [col for col in self.df.columns 
                         if col != 'MRID' and not col.endswith('_path')]
        
        result = {}
        for modality in modalities:
            try:
                path = self.get_path(mrid, modality)
                # Use normalized name as key
                canonical = self._normalize_modality(modality)
                result[canonical] = path
            except ValueError:
                # Modality doesn't exist in dataset
                try:
                    canonical = self._normalize_modality_name(modality)
                    result[canonical] = None
                except:
                    result[modality] = None
        
        return result
    
    def summary(self):
        """Print a summary of the dataset."""
        if self.df is None:
            raise ValueError("No data loaded. Run create_master_csv() or load_csv() first.")
        
        print(f"\n{'='*60}")
        print(f"Dataset Summary")
        print(f"{'='*60}")
        print(f"Total subjects (MRIDs): {len(self.df)}")
        print(f"\nModality Coverage:")
        
        coverage = self.get_modality_coverage()
        for modality, stats in sorted(coverage.items()):
            print(f"  {modality:15s}: {stats['count']:4d} ({stats['percentage']:5.1f}%)")
        
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Initialize parser with custom synonyms
    parser = NiftiMRIDParser(
        modality_synonyms={
            'FLAIR': ['FLAIR', 'FL', 'T2FLAIR'],
            'T1': ['T1', 'T1W'],
            'T1C': ['T1C', 'T1CE', 'T1POST'],
            'T2': ['T2', 'T2W'],
        }
    )
    
    # Define directory structure
    # Note: Keys can use synonyms (e.g., 'FL' instead of 'FLAIR')
    dirs = {
        'T1': '/data/uploads/t1',
        'FL': '/data/uploads/flair',  # Using synonym 'FL'
        'DLMUSE': '/data/uploads/dlmuse_seg'
    }
    
    # Create master CSV
    df = parser.create_master_csv(dirs, 'subject_index.csv')
    
    # Print summary
    parser.summary()
    
    # Query using synonyms - 'fl' and 'FLAIR' are equivalent
    result = parser.filter_by_modalities(['t1', 'fl', 'dlmuse'])
    print(f"\nSubjects with T1, FL (FLAIR), and DLMUSE: {result['count']}")
    print(f"MRIDs: {result['mrids'][:5]}...")
    
    # Query 2: Compare different modality requirements (mixing synonyms)
    comparison = parser.compare_modality_sets(
        ['t1'],
        ['t1', 'flair'],  # Using 'flair'
        ['t1', 'fl', 'dlmuse']  # Using 'fl' synonym
    )
    
    print("\nModality Set Comparison (with synonyms):")
    for item in comparison:
        mods = ', '.join(item['modalities'])
        print(f"  {mods:30s}: {item['count']:4d} subjects")
    
    # Query 3: Get specific file path using synonym
    if result['mrids']:
        mrid = result['mrids'][0]
        flair_path = parser.get_path(mrid, 'fl')  # Using 'fl' instead of 'FLAIR'
        print(f"\nFLAIR path for {mrid} (queried with 'fl'): {flair_path}")