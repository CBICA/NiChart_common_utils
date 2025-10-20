"""
Test setup script for NIFTI MRID Parser
Generates dummy .nii.gz files for testing on Windows

Tests:
1. Basic MRID extraction from various naming conventions
2. Modality synonym handling (FL <-> FLAIR, T1W <-> T1, etc.)
3. Case-insensitive queries
4. Same filename in different directories (disambiguated by directory)
5. Derived file handling (e.g., Subject001_T1_LPS_DLMUSE.nii.gz)
"""

import os
from pathlib import Path
import pandas as pd
import gzip
import random

def create_dummy_nifti(filepath):
    """Create a dummy .nii.gz file with minimal NIFTI header."""
    # Minimal NIFTI-1 header (348 bytes)
    header = bytearray(348)
    header[0:4] = (348).to_bytes(4, 'little')  # sizeof_hdr
    header[344:348] = b'n+1\0'  # magic string for .nii files
    
    # Add some dummy data (10x10x10 volume)
    data = bytes([random.randint(0, 255) for _ in range(1000)])
    
    # Write as gzipped file
    with gzip.open(filepath, 'wb') as f:
        f.write(header + data)

def setup_test_environment(base_path='./test_nifti_data'):
    """
    Create test directory structure with dummy NIFTI files.
    
    Tests multiple scenarios:
    - Multiple subjects with different naming conventions
    - Some subjects missing certain modalities
    - Derived/processed files (DLMUSE)
    - Synonym testing (FL vs FLAIR)
    - Same filename in different directories
    - Case variations
    """
    base = Path(base_path)
    
    # Create directory structure with SYNONYM directories
    dirs = {
        'T1': base / 't1',
        'T1W': base / 't1w',          # Synonym directory for T1
        'FLAIR': base / 'flair',
        'FL': base / 'fl',            # Synonym directory for FLAIR
        'T2': base / 't2',
        'DLMUSE': base / 'dlmuse_seg'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Define test subjects with various naming conventions
    test_subjects = [
        {
            'mrid': 'Subject001',
            'files': {
                'T1': 'Subject001_T1.nii.gz',
                'FLAIR': 'Subject001_FLAIR.nii.gz',
                'T2': 'Subject001_T2.nii.gz',
                'DLMUSE': 'Subject001_DLMUSE.nii.gz'
            }
        },
        {
            'mrid': 'Subject002',
            'files': {
                'T1': 'Subject002_T1.nii.gz',
                'FL': 'Subject002_FL.nii.gz',  # Using FL synonym
                'DLMUSE': 'Subject002_DLMUSE.nii.gz'
            }
        },
        {
            'mrid': 'PatientA',
            'files': {
                'T1W': 'PatientA_T1W.nii.gz',  # Using T1W synonym
                'FLAIR': 'PatientA-FLAIR.nii.gz',  # Dash separator
                'T2': 'PatientA_T2.nii.gz'
            }
        },
        {
            'mrid': 'PatientB',
            'files': {
                'T1': 'PatientB_t1.nii.gz',     # Lowercase modality
                'FLAIR': 'PatientB_flair.nii.gz'  # Lowercase modality
            }
        },
        {
            'mrid': '12345',  # Numeric ID
            'files': {
                'T1': '12345_T1.nii.gz',
                'FLAIR': '12345_FLAIR.nii.gz',
                'T2': '12345_T2.nii.gz',
                'DLMUSE': '12345_DLMUSE.nii.gz'
            }
        },
        {
            'mrid': 'ABC_XYZ',  # ID with separator
            'files': {
                'T1': 'ABC_XYZ_T1.nii.gz',
                'FL': 'ABC_XYZ_FL.nii.gz',  # Using FL synonym
                'DLMUSE': 'ABC_XYZ_DLMUSE.nii.gz'
            }
        },
        {
            'mrid': 'Study01_Sub001',
            'files': {
                'T1': 'Study01_Sub001_T1.nii.gz',
                'FLAIR': 'Study01_Sub001_FLAIR.nii.gz',
                'T2': 'Study01_Sub001_T2.nii.gz'
            }
        },
        # TEST: Same filename in different directories
        {
            'mrid': 'SameFilename',
            'files': {
                'T1': 'SameFilename.nii.gz',       # Same filename!
                'FLAIR': 'SameFilename.nii.gz',    # Same filename!
                'T2': 'SameFilename.nii.gz'        # Same filename!
            }
        }
    ]
    
    # Add derived file examples (segmentation outputs from T1)
    derived_examples = [
        ('Subject001', 'DLMUSE', 'Subject001_T1_LPS_DLMUSE.nii.gz'),
        ('Subject002', 'DLMUSE', 'Subject002_T1_LPS_DLMUSE.nii.gz'),
        ('12345', 'DLMUSE', '12345_T1_LPS_DLMUSE.nii.gz'),
    ]
    
    created_files = []
    
    # Create files for each subject
    print(f"\n{'='*60}")
    print("CREATING TEST DATA")
    print(f"{'='*60}\n")
    
    for subject in test_subjects:
        mrid = subject['mrid']
        print(f"Creating files for MRID: {mrid}")
        
        for modality, filename in subject['files'].items():
            # Get the directory for this modality
            if modality in dirs:
                target_dir = dirs[modality]
            else:
                continue
            
            filepath = target_dir / filename
            create_dummy_nifti(filepath)
            created_files.append(str(filepath))
            print(f"  ✓ {modality:10s}: {filename} -> {target_dir.name}/")
    
    # Create derived files in DLMUSE directory
    print(f"\nCreating derived (processed) files:")
    for mrid, modality, filename in derived_examples:
        filepath = dirs[modality] / filename
        if not filepath.exists():  # Don't overwrite if already created
            create_dummy_nifti(filepath)
            created_files.append(str(filepath))
            print(f"  ✓ {mrid}: {filename}")
    
    print(f"\n{'='*60}")
    print(f"Test environment created at: {base.absolute()}")
    print(f"Total files created: {len(created_files)}")
    print(f"{'='*60}\n")
    
    return {
        'base_path': str(base.absolute()),
        'directories': {k: str(v) for k, v in dirs.items()},
        'created_files': created_files
    }


def test_mrid_extraction(parser):
    """Test MRID extraction from various filename patterns."""
    print("\n" + "="*60)
    print("TEST 1: MRID EXTRACTION FROM FILENAMES")
    print("="*60)
    
    test_cases = [
        ('Subject001_T1.nii.gz', 'Subject001'),
        ('PatientA-FLAIR.nii', 'PatientA'),
        ('12345_T1_LPS_DLMUSE.nii.gz', '12345'),  # Derived file
        ('Study01_Sub001_t2.nii.gz', 'Study01_Sub001'),
        ('Subject002_FL.nii.gz', 'Subject002'),  # FL synonym
        ('ABC_XYZ_T1.nii.gz', 'ABC_XYZ'),
        ('scan.nii.gz', 'scan'),  # No modality in filename
        ('Subject001_T1_LPS_DLMUSE.nii.gz', 'Subject001'),  # Complex derived
    ]
    
    print("\nTesting that MRID extraction correctly handles various patterns:")
    all_passed = True
    for filename, expected_mrid in test_cases:
        extracted = parser.extract_mrid(filename)
        status = "✓ PASS" if extracted == expected_mrid else "✗ FAIL"
        if extracted != expected_mrid:
            all_passed = False
        print(f"  {status}: {filename:40s} -> {extracted:20s} (expected: {expected_mrid})")
    
    if all_passed:
        print("\n✓ All MRID extraction tests passed!")
    else:
        print("\n✗ Some MRID extraction tests failed. Check parsing logic.")
    
    return all_passed


def test_synonym_handling(parser):
    """Test that modality synonyms are handled correctly."""
    print("\n" + "="*60)
    print("TEST 2: MODALITY SYNONYM HANDLING")
    print("="*60)
    
    if parser.df is None:
        print("✗ No data loaded. Cannot test synonyms.")
        return False
    
    print("\nTesting that FL and FLAIR are treated as the same modality:")
    print(f"  Configured synonyms: {parser.modality_synonyms.get('FLAIR', [])}")
    
    # Check that FL and FLAIR queries return the same results
    try:
        # Query with 'FLAIR'
        result_flair = parser.filter_by_modalities(['T1', 'FLAIR'])
        print(f"\n  Query with ['T1', 'FLAIR']: {result_flair['count']} subjects")
        
        # Query with 'FL' (synonym)
        result_fl = parser.filter_by_modalities(['T1', 'FL'])
        print(f"  Query with ['T1', 'FL']:    {result_fl['count']} subjects")
        
        # They should return the same subjects
        if result_flair['mrids'] == result_fl['mrids']:
            print(f"\n  ✓ PASS: Both queries returned identical subject lists")
            print(f"  Common subjects: {result_flair['mrids'][:3]}...")
            
            # Test case insensitivity too
            result_mixed = parser.filter_by_modalities(['t1', 'flair'])
            if result_mixed['mrids'] == result_flair['mrids']:
                print(f"  ✓ PASS: Case-insensitive queries also work (tested with ['t1', 'flair'])")
                return True
            else:
                print(f"  ✗ FAIL: Case-insensitive query returned different results")
                return False
        else:
            print(f"\n  ✗ FAIL: Queries returned different subjects")
            print(f"  FLAIR only: {set(result_flair['mrids']) - set(result_fl['mrids'])}")
            print(f"  FL only: {set(result_fl['mrids']) - set(result_flair['mrids'])}")
            return False
            
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def test_same_filename_different_dirs(parser):
    """Test handling of same filename in different directories."""
    print("\n" + "="*60)
    print("TEST 3: SAME FILENAME IN DIFFERENT DIRECTORIES")
    print("="*60)
    
    if parser.df is None:
        print("✗ No data loaded. Cannot test.")
        return False
    
    print("\nTesting that 'scan.nii.gz' in different directories is disambiguated:")
    
    # Check if SameFilename subject exists
    same_filename_row = parser.df[parser.df['MRID'] == 'SameFilename']
    
    if same_filename_row.empty:
        print("  ✗ FAIL: 'SameFilename' subject not found in dataset")
        print("  Troubleshooting: Check that scan.nii.gz files were created in T1/, FLAIR/, T2/ dirs")
        return False
    
    # Check if it has multiple modalities
    modalities_found = []
    for col in ['T1', 'FLAIR', 'T2']:
        if col in parser.df.columns and pd.notna(same_filename_row[col].iloc[0]):
            modalities_found.append(col)
    
    print(f"  Subject 'SameFilename' found with modalities: {modalities_found}")
    
    if len(modalities_found) >= 2:
        print(f"  ✓ PASS: Same filename successfully disambiguated by directory")
        
        # Show the paths
        for mod in modalities_found:
            try:
                path = parser.get_path('SameFilename', mod)
                print(f"    {mod:10s}: {path}")
            except:
                pass
        return True
    else:
        print(f"  ✗ FAIL: Only {len(modalities_found)} modality found, expected 2+")
        print("  Troubleshooting: Check that files were created in multiple directories")
        return False


def test_derived_files(parser):
    """Test that derived files don't create spurious subjects."""
    print("\n" + "="*60)
    print("TEST 4: DERIVED FILE HANDLING")
    print("="*60)
    
    if parser.df is None:
        print("✗ No data loaded. Cannot test.")
        return False
    
    print("\nTesting that derived files like 'Subject001_T1_LPS_DLMUSE.nii.gz' don't create false MRIDs:")
    
    # Check for spurious MRIDs that might have been created from derived files
    spurious_mrids = [
        'Subject001LPS',
        'Subject001_LPS',
        'Subject002LPS',
        '12345LPS'
    ]
    
    found_spurious = []
    for spurious in spurious_mrids:
        if spurious in parser.df['MRID'].values:
            found_spurious.append(spurious)
    
    if found_spurious:
        print(f"  ✗ FAIL: Found spurious MRIDs from derived files: {found_spurious}")
        print("  Troubleshooting: Check MRID extraction logic - it should strip from FIRST modality keyword")
        return False
    else:
        print(f"  ✓ PASS: No spurious MRIDs found")
        
        # Verify the correct MRIDs exist
        expected_mrids = ['Subject001', 'Subject002', '12345']
        found_expected = [m for m in expected_mrids if m in parser.df['MRID'].values]
        
        print(f"  ✓ Correct base MRIDs found: {found_expected}")
        
        # Check that derived files are associated with correct MRIDs
        for mrid in found_expected:
            dlmuse_files = []
            row = parser.df[parser.df['MRID'] == mrid]
            if not row.empty and 'DLMUSE' in parser.df.columns:
                filename = row['DLMUSE'].iloc[0]
                if pd.notna(filename):
                    dlmuse_files.append(filename)
            if dlmuse_files:
                print(f"    {mrid}: {dlmuse_files}")
        
        return True


def run_comprehensive_tests(parser):
    """Run all test suites."""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Test 1: MRID extraction
    test1_passed = test_mrid_extraction(parser)
    
    # Import pandas for tests that need it
    try:
        import pandas as pd
        globals()['pd'] = pd
    except ImportError:
        print("\n✗ pandas not installed. Skipping DataFrame-dependent tests.")
        return
    
    # Test 2: Synonym handling
    test2_passed = test_synonym_handling(parser)
    
    # Test 3: Same filename, different directories
    test3_passed = test_same_filename_different_dirs(parser)
    
    # Test 4: Derived files
    test4_passed = test_derived_files(parser)
    
    # Test 5: Standard queries (from before)
    print("\n" + "="*60)
    print("TEST 5: STANDARD QUERY OPERATIONS")
    print("="*60)
    test5_passed = run_standard_queries(parser)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    tests = [
        ("MRID Extraction", test1_passed),
        ("Synonym Handling", test2_passed),
        ("Same Filename Disambiguation", test3_passed),
        ("Derived File Handling", test4_passed),
        ("Standard Queries", test5_passed)
    ]
    
    for test_name, passed in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(passed for _, passed in tests)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review output above for details")
    
    print("="*60 + "\n")


def run_standard_queries(parser):
    """Run example queries to demonstrate functionality."""
    if parser.df is None:
        print("✗ No data loaded. Cannot run queries.")
        return False
    
    try:
        # Summary
        parser.summary()
        
        # Query 1: Subjects with complete data
        print("\n1. Subjects with T1, FLAIR, T2, and DLMUSE:")
        result = parser.filter_by_modalities(['T1', 'FLAIR', 'T2', 'DLMUSE'])
        print(f"   Count: {result['count']}")
        if result['mrids']:
            print(f"   MRIDs: {result['mrids']}")
        
        # Query 2: Using synonyms
        print("\n2. Subjects with T1 and FL (FLAIR synonym):")
        result = parser.filter_by_modalities(['T1', 'FL'])
        print(f"   Count: {result['count']}")
        if result['mrids']:
            print(f"   MRIDs: {result['mrids'][:5]}{'...' if len(result['mrids']) > 5 else ''}")
        
        # Query 3: Case-insensitive queries
        print("\n3. Case-insensitive query with ['t1', 'flair'] (lowercase):")
        result = parser.filter_by_modalities(['t1', 'flair'])
        print(f"   Count: {result['count']}")
        
        # Query 4: Compare different requirements
        print("\n4. Comparison of modality requirements:")
        comparison = parser.compare_modality_sets(
            ['T1'],
            ['T1', 'FLAIR'],
            ['T1', 'FL'],  # Should match T1+FLAIR
            ['T1', 'FLAIR', 'T2']
        )
        for item in comparison:
            mods = ', '.join(item['modalities'])
            print(f"   {mods:30s}: {item['count']:2d} subjects")
        
        # Query 5: Get paths
        if result['mrids']:
            mrid = result['mrids'][0]
            print(f"\n5. Paths for subject '{mrid}':")
            paths = parser.get_paths(mrid)
            for modality, path in paths.items():
                if path:
                    print(f"   {modality:10s}: {path}")
                else:
                    print(f"   {modality:10s}: [MISSING]")
            
            # Test synonym path retrieval
            print(f"\n6. Testing synonym path retrieval for '{mrid}':")
            try:
                flair_path = parser.get_path(mrid, 'FLAIR')
                fl_path = parser.get_path(mrid, 'FL')
                if flair_path == fl_path:
                    print(f"   ✓ FLAIR and FL return same path: {flair_path}")
                else:
                    print(f"   ✗ FLAIR and FL return different paths!")
            except Exception as e:
                print(f"   Note: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR during standard queries: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Step 1: Create test data
    print("="*60)
    print("NIFTI MRID PARSER - COMPREHENSIVE TEST SUITE")
    print("="*60)
    print("\nSetting up test environment with:")
    print("  - Multiple naming conventions")
    print("  - Modality synonyms (FL/FLAIR, T1/T1W)")
    print("  - Same filename in different directories")
    print("  - Derived files (e.g., _T1_LPS_DLMUSE)")
    print("  - Case variations")
    
    test_info = setup_test_environment('./test_nifti_data')
    
    # Step 2: Import and use the parser
    print("\nImporting NiftiMRIDParser...")
    
    try:
        from NiChart_common_utils.nifti_parser import NiftiMRIDParser
    except ImportError:
        print("\n" + "="*60)
        print("ERROR: Could not import NiftiMRIDParser")
        print("="*60)
        print("\nPlease save the NiftiMRIDParser class in a file called 'nifti_parser.py'")
        print("in the same directory as this script.")
        print(f"\nTest data has been created at: {test_info['base_path']}")
        print("\nDirectories created:")
        for mod, path in test_info['directories'].items():
            print(f"  {mod:10s}: {path}")
        exit(1)
    
    # Step 3: Parse the test data with SYNONYM directories
    print("\nParsing NIFTI files...")
    parser = NiftiMRIDParser()
    
    # Use multiple directories including synonyms
    dirs_to_scan = {
        'T1': test_info['directories']['T1'],
        'T1W': test_info['directories']['T1W'],      # Synonym directory
        'FLAIR': test_info['directories']['FLAIR'],
        'FL': test_info['directories']['FL'],        # Synonym directory
        'T2': test_info['directories']['T2'],
        'DLMUSE': test_info['directories']['DLMUSE']
    }
    
    df = parser.create_master_csv(dirs_to_scan, 'test_master_index.csv')
    
    # Step 4: Run comprehensive test suite
    run_comprehensive_tests(parser)
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
    print(f"Master CSV created: test_master_index.csv")
    print(f"Test data location: {test_info['base_path']}")
    print("\nYou can inspect the CSV to see how different naming conventions")
    print("and synonyms are normalized in the dataset.")
    print("="*60 + "\n")