import os
import glob
import shutil
import argparse
import pandas as pd
import numpy as np
import SimpleITK as sitk
from skimage.filters import frangi
from skimage.measure import label, regionprops
import multiprocessing

def setup_args():
    parser = argparse.ArgumentParser(description="Phase 1: LUNA16 Extraction and Frangi Pseudo-Labeling")
    parser.add_argument("--luna-dir", type=str, default="./data/luna16", help="Path to LUNA16 dataset root")
    parser.add_argument("--output-dir", type=str, default="./output/phase1_data", help="Where to save the tensors")
    parser.add_argument("--num-img", type=int, default=1, help="Number of annotations to process (0 for ALL images)")
    parser.add_argument("--frangi-thresh", type=float, default=0.005, help="Binary threshold for Frangi output")
    parser.add_argument("--cca-thresh", type=int, default=75, help="Minimum voxel count for CCA islands")
    parser.add_argument("--cores", type=int, default=10, help="Number of CPU cores to use for parallel processing")
    parser.add_argument("--patch-size", type=int, default=32, help="Cubic patch size in voxels (spacing stays 1mm/voxel)")
    return parser.parse_args()

def extract_resampled_crop(image, center_physical, output_size=(32, 32, 32), output_spacing=(1.0, 1.0, 1.0)):
    """
    Directly extracts and resamples a specific physical volume from the CT scan.
    """
    # Calculate the bottom-left-back physical coordinate of our 32x32x32mm box
    origin = [c - (s * sp) / 2.0 for c, s, sp in zip(center_physical, output_size, output_spacing)]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(output_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(origin)
    resample.SetTransform(sitk.Transform()) # Identity transform
    resample.SetDefaultPixelValue(-1000)    # Background = Air (Hounsfield Units)
    resample.SetInterpolator(sitk.sitkBSpline) # High quality interpolation for CT
    
    cropped_sitk = resample.Execute(image)
    return sitk.GetArrayFromImage(cropped_sitk) # Returns shape (Z, Y, X)

def apply_frangi_and_sanitize(ct_patch, frangi_thresh, cca_thresh):
    """
    Applies the Frangi filter and uses CCA to clean up noise.
    """
    # Normalize CT patch to [0, 1] for the Frangi filter
    clipped_ct = np.clip(ct_patch, -1000, 400)
    normalized_ct = (clipped_ct + 1000) / 1400.0

    # Frangi Filter
    vesselness = frangi(normalized_ct, sigmas=range(1, 3, 1), black_ridges=False)
    
    # Create initial Binary Mask
    binary_mask = vesselness > frangi_thresh

    # Sanitization via CCA
    labeled_mask = label(binary_mask, connectivity=3)
    clean_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    
    for region in regionprops(labeled_mask):
        if region.area >= cca_thresh:
            # Reconstruct the cleaned mask
            coords = region.coords
            clean_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
            
    return normalized_ct, clean_mask

def worker_process(task_args):
    """
    The function executed by each parallel worker. 
    It unpacks the task arguments and does the heavy ITK/Frangi lifting.
    """
    # Restrict SimpleITK to 1 thread per process to prevent core contention
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    
    uid, center_physical, mhd_path, output_dir, frangi_thresh, cca_thresh, processed_idx, patch_size = task_args

    try:
        # Load the CT metadata and image
        itk_image = sitk.ReadImage(mhd_path)

        # Extract and Resample
        ct_patch = extract_resampled_crop(itk_image, center_physical, output_size=(patch_size, patch_size, patch_size))
        
        # Pseudo-labeling and Sanitization
        normalized_ct, clean_mask = apply_frangi_and_sanitize(ct_patch, frangi_thresh, cca_thresh)
        
        # Save Tensors
        save_prefix = os.path.join(output_dir, f"patch_{processed_idx:04d}_{uid}")
        np.save(f"{save_prefix}_ct.npy", normalized_ct.astype(np.float32))
        np.save(f"{save_prefix}_mask.npy", clean_mask.astype(np.float32))
        
        return True, uid
    except Exception as e:
        return False, f"Failed on {uid}: {str(e)}"

def main():
    args = setup_args()
    
    # Overwrite logic: clear the directory if it exists, then recreate
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    
    # Load Annotations
    ann_path = os.path.join(args.luna_dir, "annotations.csv")
    df = pd.read_csv(ann_path)
    
    tasks = []
    processed_count = 0
    target_count = len(df) if args.num_img == 0 else args.num_img
    
    print(f"Phase 1: Discovering files for {target_count} target nodule(s)...")
    
    # Fast Sequential Discovery
    for idx, row in df.iterrows():
        if args.num_img != 0 and processed_count >= args.num_img:
            break
            
        uid = row['seriesuid']
        center_physical = (row['coordX'], row['coordY'], row['coordZ'])
        
        # Search for the .mhd file in subset folders
        search_pattern = os.path.join(args.luna_dir, "subset*", "**", f"{uid}.mhd")
        mhd_files = glob.glob(search_pattern, recursive=True)
        
        if not mhd_files:
            continue
            
        mhd_path = mhd_files[0]
        
        # Package the job
        tasks.append((
            uid, center_physical, mhd_path,
            args.output_dir, args.frangi_thresh, args.cca_thresh, processed_count, args.patch_size
        ))
        
        processed_count += 1

    actual_count = len(tasks)
    if actual_count == 0:
        print("No valid files found to process. Check your dataset paths.")
        return

    print(f"Found {actual_count} valid files. Processing across {args.cores} core(s)...")

    # Parallel Execution
    successful_count = 0
    with multiprocessing.Pool(processes=args.cores) as pool:
        # pool.imap_unordered yields results as soon as they are ready
        for idx, (success, msg) in enumerate(pool.imap_unordered(worker_process, tasks)):
            if success:
                successful_count += 1
                # Print progress 
                print(f"Progress: [{idx + 1}/{actual_count}] completed.", end='\r')
            else:
                print(f"\nError: {msg}")
                
    print(f"\nSuccess! Successfully processed {successful_count}/{actual_count} patches. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()