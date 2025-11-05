"""
Dataset Preparation Script for AgeBooth from wiki.mat

This script reads the wiki.mat file and organizes images into:
- Training data: young (10-20 years) and old (70-80 years)
- Validation data: young (10-20 years) and old (70-80 years)

Requirements:
    pip install scipy pillow numpy opencv-python tqdm
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from scipy.io import loadmat
from PIL import Image
import cv2
from tqdm import tqdm
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
WIKI_CROP_BASE = Path("./wiki_crop/wiki_crop")  # Base path for wiki_crop
WIKI_MAT_PATH = WIKI_CROP_BASE / "wiki.mat"
OUTPUT_BASE = Path("./dataset")

# Age groups to extract
AGE_GROUPS = {
    "young_10_20": (10, 20),
    "old_70_80": (70, 80),
}

# Dataset split
MIN_TRAIN_IMAGES = 25  # Minimum images per age group for training
MIN_VAL_IMAGES = 25    # Minimum images per age group for validation
TRAIN_VAL_RATIO = 0.8  # 80% training, 20% validation

# Quality filters
MIN_FACE_SCORE = 1.0           # Minimum face detection confidence
MAX_SECOND_FACE_SCORE = 0.5    # Maximum second face score (to avoid multiple faces)
MIN_IMAGE_SIZE = 256           # Minimum image width/height
MIN_FACE_SIZE = 100            # Minimum face width/height

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def matlab_datenum_to_age(matlab_datenum, photo_taken_year):
    """
    Convert Matlab datenum to age.
    
    Args:
        matlab_datenum: Matlab serial date number for date of birth
        photo_taken_year: Year when photo was taken
        
    Returns:
        Age in years (int) or None if invalid
    """
    try:
        # Matlab datenum starts from January 1, 0000
        # Python datetime starts from January 1, 1
        # Offset is 366 days (year 0 was a leap year in Matlab)
        
        # Convert Matlab datenum to Python datetime
        python_datetime = datetime.fromordinal(int(matlab_datenum)) - timedelta(days=366)
        birth_year = python_datetime.year
        
        # Calculate age
        age = photo_taken_year - birth_year
        
        # Sanity check: age should be between 0 and 120
        if 0 <= age <= 120:
            return age
        else:
            return None
            
    except (ValueError, OverflowError, OSError):
        return None


def parse_face_location(face_location):
    """
    Parse face location array.
    
    Args:
        face_location: Array with [x1, y1, x2, y2] coordinates
        
    Returns:
        Tuple (x1, y1, x2, y2) or None if invalid
    """
    try:
        if len(face_location) == 4:
            x1, y1, x2, y2 = face_location
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure valid coordinates
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)
    except (TypeError, ValueError):
        pass
    
    return None


def check_image_quality(img_path, face_location=None):
    """
    Check if image meets quality requirements.
    
    Args:
        img_path: Path to image file
        face_location: Optional face bounding box (x1, y1, x2, y2)
        
    Returns:
        bool: True if image passes quality checks
    """
    try:
        # Check if file exists
        if not img_path.exists():
            return False
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        
        height, width = img.shape[:2]
        
        # Check minimum image size
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            return False
        
        # Check face size if face_location provided
        if face_location:
            x1, y1, x2, y2 = face_location
            face_width = x2 - x1
            face_height = y2 - y1
            
            if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                return False
            
            # Check if face location is within image bounds
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                return False
        
        return True
        
    except Exception as e:
        print(f"Error checking image quality for {img_path}: {e}")
        return False


def crop_and_save_face(img_path, face_location, output_path):
    """
    Crop face from image and save to output path.
    
    Args:
        img_path: Path to source image
        face_location: Face bounding box (x1, y1, x2, y2)
        output_path: Path to save cropped face
        
    Returns:
        bool: True if successful
    """
    try:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        
        # Crop face
        x1, y1, x2, y2 = face_location
        face_img = img[y1:y2, x1:x2]
        
        # Resize to standard size if too large
        height, width = face_img.shape[:2]
        if max(height, width) > 512:
            scale = 512 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            face_img = cv2.resize(face_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Save cropped face
        cv2.imwrite(str(output_path), face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True
        
    except Exception as e:
        print(f"Error cropping face from {img_path}: {e}")
        return False


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_wiki_mat():
    """
    Main function to process wiki.mat and organize dataset.
    """
    
    print("="*80)
    print("AgeBooth Dataset Preparation from wiki.mat")
    print("="*80)
    print()
    
    # Check if wiki.mat exists
    if not WIKI_MAT_PATH.exists():
        print(f"ERROR: wiki.mat not found at {WIKI_MAT_PATH}")
        print(f"Please make sure the file exists at the correct location.")
        return
    
    print(f"Loading wiki.mat from: {WIKI_MAT_PATH}")
    
    # Load .mat file
    try:
        mat_data = loadmat(str(WIKI_MAT_PATH))
    except Exception as e:
        print(f"ERROR: Failed to load wiki.mat: {e}")
        return
    
    # Extract data from nested structure
    # The data is typically stored in mat_data['wiki'][0, 0]
    if 'wiki' in mat_data:
        wiki = mat_data['wiki'][0, 0]
    elif 'imdb' in mat_data:
        wiki = mat_data['imdb'][0, 0]
    else:
        print(f"ERROR: Could not find 'wiki' or 'imdb' key in mat file")
        print(f"Available keys: {list(mat_data.keys())}")
        return
    
    # Extract fields
    dob = wiki['dob'][0]                          # Date of birth (Matlab datenum)
    photo_taken = wiki['photo_taken'][0]          # Year photo was taken
    full_path = wiki['full_path'][0]              # Path to image
    gender = wiki['gender'][0]                    # 0=female, 1=male, NaN=unknown
    face_location = wiki['face_location'][0]      # Face bounding box
    face_score = wiki['face_score'][0]            # Face detection score
    second_face_score = wiki['second_face_score'][0]  # Second face score
    
    num_images = len(dob)
    print(f"Found {num_images} images in wiki.mat")
    print()
    
    # Create output directory structure
    print("Creating output directory structure...")
    
    train_dirs = {}
    val_dirs = {}
    
    for age_group in AGE_GROUPS.keys():
        train_dir = OUTPUT_BASE / "training" / age_group
        val_dir = OUTPUT_BASE / "validation" / age_group
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        train_dirs[age_group] = train_dir
        val_dirs[age_group] = val_dir
    
    print(f"Output directory: {OUTPUT_BASE}")
    print()
    
    # Collect valid images for each age group
    print("Filtering images by age and quality...")
    
    age_group_images = {group: [] for group in AGE_GROUPS.keys()}
    
    stats = {
        'total_processed': 0,
        'invalid_age': 0,
        'low_face_score': 0,
        'multiple_faces': 0,
        'poor_quality': 0,
        'file_not_found': 0,
    }
    
    for i in tqdm(range(num_images), desc="Processing images"):
        stats['total_processed'] += 1
        
        # Extract data for this image
        try:
            img_dob = dob[i]
            img_year = int(photo_taken[i])
            img_path_str = full_path[i][0] if isinstance(full_path[i], np.ndarray) else str(full_path[i])
            img_gender = gender[i]
            img_face_loc = face_location[i]
            img_face_score = face_score[i]
            img_second_face_score = second_face_score[i]
        except (IndexError, ValueError) as e:
            continue
        
        # Calculate age
        age = matlab_datenum_to_age(img_dob, img_year)
        if age is None:
            stats['invalid_age'] += 1
            continue
        
        # Check face score
        if np.isinf(img_face_score) or img_face_score < MIN_FACE_SCORE:
            stats['low_face_score'] += 1
            continue
        
        # Check for multiple faces
        if not np.isnan(img_second_face_score) and img_second_face_score > MAX_SECOND_FACE_SCORE:
            stats['multiple_faces'] += 1
            continue
        
        # Construct full image path
        img_full_path = WIKI_CROP_BASE / img_path_str
        
        # Check if file exists
        if not img_full_path.exists():
            stats['file_not_found'] += 1
            continue
        
        # Parse face location
        face_bbox = parse_face_location(img_face_loc)
        
        # Check image quality
        if not check_image_quality(img_full_path, face_bbox):
            stats['poor_quality'] += 1
            continue
        
        # Check if age falls into any of our target groups
        for group_name, (min_age, max_age) in AGE_GROUPS.items():
            if min_age <= age <= max_age:
                age_group_images[group_name].append({
                    'path': img_full_path,
                    'age': age,
                    'gender': img_gender,
                    'face_location': face_bbox,
                    'face_score': img_face_score,
                })
                break
    
    print()
    print("Filtering Statistics:")
    print("-" * 40)
    print(f"Total images processed: {stats['total_processed']}")
    print(f"Invalid age: {stats['invalid_age']}")
    print(f"Low face score: {stats['low_face_score']}")
    print(f"Multiple faces: {stats['multiple_faces']}")
    print(f"Poor quality: {stats['poor_quality']}")
    print(f"File not found: {stats['file_not_found']}")
    print()
    
    print("Images found per age group:")
    print("-" * 40)
    for group_name, images in age_group_images.items():
        print(f"{group_name}: {len(images)} images")
    print()
    
    # Split into training and validation sets
    print("Splitting into training and validation sets...")
    
    final_stats = {
        'train': {},
        'val': {}
    }
    
    for group_name, images in age_group_images.items():
        if len(images) == 0:
            print(f"WARNING: No images found for {group_name}")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split point
        total_needed = MIN_TRAIN_IMAGES + MIN_VAL_IMAGES
        
        if len(images) < total_needed:
            print(f"WARNING: Only {len(images)} images for {group_name} (need {total_needed})")
            train_images = images[:MIN_TRAIN_IMAGES] if len(images) >= MIN_TRAIN_IMAGES else images
            val_images = images[MIN_TRAIN_IMAGES:] if len(images) > MIN_TRAIN_IMAGES else []
        else:
            # Use ratio-based split, ensuring minimum counts
            split_idx = max(MIN_TRAIN_IMAGES, int(len(images) * TRAIN_VAL_RATIO))
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Ensure we have enough validation images
            if len(val_images) < MIN_VAL_IMAGES and len(train_images) > MIN_TRAIN_IMAGES:
                extra_needed = MIN_VAL_IMAGES - len(val_images)
                val_images.extend(train_images[-extra_needed:])
                train_images = train_images[:-extra_needed]
        
        # Save training images
        train_count = 0
        for idx, img_data in enumerate(tqdm(train_images, desc=f"Saving {group_name} training images")):
            output_filename = f"{group_name}_train_{idx:04d}_age{img_data['age']}.jpg"
            output_path = train_dirs[group_name] / output_filename
            
            if img_data['face_location']:
                # Crop and save face
                if crop_and_save_face(img_data['path'], img_data['face_location'], output_path):
                    train_count += 1
            else:
                # Just copy the full image
                try:
                    shutil.copy2(img_data['path'], output_path)
                    train_count += 1
                except Exception as e:
                    print(f"Error copying {img_data['path']}: {e}")
        
        final_stats['train'][group_name] = train_count
        
        # Save validation images
        val_count = 0
        for idx, img_data in enumerate(tqdm(val_images, desc=f"Saving {group_name} validation images")):
            output_filename = f"{group_name}_val_{idx:04d}_age{img_data['age']}.jpg"
            output_path = val_dirs[group_name] / output_filename
            
            if img_data['face_location']:
                # Crop and save face
                if crop_and_save_face(img_data['path'], img_data['face_location'], output_path):
                    val_count += 1
            else:
                # Just copy the full image
                try:
                    shutil.copy2(img_data['path'], output_path)
                    val_count += 1
                except Exception as e:
                    print(f"Error copying {img_data['path']}: {e}")
        
        final_stats['val'][group_name] = val_count
    
    print()
    print("="*80)
    print("Dataset Preparation Complete!")
    print("="*80)
    print()
    print("Final Dataset Statistics:")
    print("-" * 40)
    print("Training Set:")
    for group_name, count in final_stats['train'].items():
        print(f"  {group_name}: {count} images")
        print(f"    → {train_dirs[group_name]}")
    print()
    print("Validation Set:")
    for group_name, count in final_stats['val'].items():
        print(f"  {group_name}: {count} images")
        print(f"    → {val_dirs[group_name]}")
    print()
    print("="*80)
    print()
    
    # Check if we have enough images
    success = True
    for group_name in AGE_GROUPS.keys():
        if final_stats['train'].get(group_name, 0) < MIN_TRAIN_IMAGES:
            print(f"⚠️  WARNING: Only {final_stats['train'].get(group_name, 0)} training images for {group_name} (need {MIN_TRAIN_IMAGES})")
            success = False
        if final_stats['val'].get(group_name, 0) < MIN_VAL_IMAGES:
            print(f"⚠️  WARNING: Only {final_stats['val'].get(group_name, 0)} validation images for {group_name} (need {MIN_VAL_IMAGES})")
            success = False
    
    if success:
        print("✅ Dataset prepared successfully! You can now proceed with training.")
    else:
        print("⚠️  Dataset prepared but some age groups have fewer images than required.")
        print("    You may need to adjust MIN_TRAIN_IMAGES or MIN_VAL_IMAGES in the script.")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run main processing
    process_wiki_mat()
