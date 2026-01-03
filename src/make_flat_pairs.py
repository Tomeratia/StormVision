
"""
Creates a flat paired dataset (ORIG + SYNTH) using `meta_synth.csv`.
Copies each original image and its synthetic counterpart into `paired_flat_visuals/`
with consistent names: pair_<id>_ORIG_* and pair_<id>_SYNTH_*, and saves `pairs_manifest.csv`.
"""

import shutil
import pandas as pd
from pathlib import Path




# Folder created by the generation step (contains synthetic images + metadata files)
SYNTH_BASE_DIR = Path('/Users/tomeratia/Desktop/לימודים/תואר ראשון/שנה ג/Gen Ai/StormVision/storm_synth_out_onefolder')
SYNTH_IMAGES_DIR = SYNTH_BASE_DIR / 'images'
META_CSV_PATH = SYNTH_BASE_DIR / 'meta_synth.csv'  # Links each original image to its generated synthetic image

# Original image folders (where the real images live)
ORIG_TRAIN_DIR = Path('/Users/tomeratia/Desktop/לימודים/תואר ראשון/שנה ג/Gen Ai/StormVision/data/images/train')
ORIG_VAL_DIR   = Path('/Users/tomeratia/Desktop/לימודים/תואר ראשון/שנה ג/Gen Ai/StormVision/data/images/val')
ORIG_TEST_DIR  = Path('/Users/tomeratia/Desktop/לימודים/תואר ראשון/שנה ג/Gen Ai/StormVision/data/images/test')

# Output folder: a flat paired dataset (ORIG + SYNTH) with consistent filenames
OUT_DIR = SYNTH_BASE_DIR / 'paired_flat_visuals'


def safe_copy(src: Path, dst: Path):
    """Copies a file only if it exists (returns True on success)."""
    if not src.exists():
        return False
    try:
        shutil.copy2(src, dst)  # copy2 preserves file metadata (timestamps), not strictly required but safe
        return True
    except Exception as e:
        print(f"Error copying {src.name}: {e}")
        return False

def find_original_image(filename):
    """Searches for the original image across Train/Val/Test folders."""
    # Try Train
    p = ORIG_TRAIN_DIR / filename
    if p.exists(): return p
    
    # Try Val
    p = ORIG_VAL_DIR / filename
    if p.exists(): return p
    
    # Try Test
    p = ORIG_TEST_DIR / filename
    if p.exists(): return p
    
    return None



def main():
    print("=== Starting Pair Generation ===")
    
    # Sanity checks: required inputs must exist before processing
    if not META_CSV_PATH.exists():
        print(f"CRITICAL ERROR: Could not find meta CSV at: {META_CSV_PATH}")
        print("Did you run the generation script? It creates this CSV.")
        return

    if not SYNTH_IMAGES_DIR.exists():
        print(f"CRITICAL ERROR: Synth images dir not found at: {SYNTH_IMAGES_DIR}")
        return

    # Create output directory (idempotent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory created at: {OUT_DIR}")

    # Load metadata mapping: orig_file -> out_file
    df = pd.read_csv(META_CSV_PATH)
    print(f"Loaded CSV with {len(df)} rows.")

    manifest_rows = []
    count_ok = 0
    count_missing = 0

    # Iterate rows: for each original image, locate its synthetic counterpart and copy both into OUT_DIR
    for i, row in df.iterrows():
        orig_filename = row['orig_file']
        synth_filename = row['out_file']
        
        # Resolve full paths (original is searched across train/val/test; synth is under SYNTH_IMAGES_DIR)
        orig_path = find_original_image(orig_filename)
        synth_path = SYNTH_IMAGES_DIR / synth_filename
        
        if orig_path is None:
            print(f"Skipping: Original image not found anywhere: {orig_filename}")
            count_missing += 1
            continue
            
        if not synth_path.exists():
            print(f"Skipping: Synth image missing: {synth_filename}")
            count_missing += 1
            continue

        # Create a stable pair id (used to keep ORIG/SYNTH together later when splitting to avoid leakage)
        pair_id = f"{i+1:04d}" # 0001, 0002...
        
        # Flat naming scheme: pair_<id>_ORIG_<orig_filename>
        dst_orig_name = f"pair_{pair_id}_ORIG_{orig_filename}"
        dst_orig_path = OUT_DIR / dst_orig_name
        
        # Flat naming scheme: pair_<id>_SYNTH_<synth_filename>
        dst_synth_name = f"pair_{pair_id}_SYNTH_{synth_filename}"
        dst_synth_path = OUT_DIR / dst_synth_name

        # Copy files into the flat paired folder
        safe_copy(orig_path, dst_orig_path)
        safe_copy(synth_path, dst_synth_path)

        # Record the new paired paths into a manifest (useful for later analysis and reproducible dataset splits)
        manifest_rows.append({
            "pair_id": pair_id,
            "orig_abs_path": str(dst_orig_path),
            "synth_abs_path": str(dst_synth_path),
            "scenario": row.get('scenario', 'unknown')  # optional column in meta CSV (fallback to 'unknown')
        })
        
        count_ok += 1
        if count_ok % 10 == 0:
            print(f"Processed {count_ok} pairs...", end='\r')

    # Save manifest CSV describing all successfully created pairs
    out_manifest_path = OUT_DIR / "pairs_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(out_manifest_path, index=False)

    print("\n" + "="*40)
    print(f"DONE!")
    print(f"Successfully paired: {count_ok}")
    print(f"Missing/Skipped:     {count_missing}")
    print(f"Files saved to:      {OUT_DIR}")
    print(f"Manifest saved to:   {out_manifest_path}")
    print("="*40)

if __name__ == "__main__":
    main()
