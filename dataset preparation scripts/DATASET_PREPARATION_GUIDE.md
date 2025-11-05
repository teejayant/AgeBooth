# Dataset Preparation from wiki.mat

## 📖 Overview

This script processes the `wiki.mat` file from the IMDB-WIKI dataset and organizes face images into training and validation sets for two age groups:
- **Young (10-20 years old)**
- **Old (70-80 years old)**

## 🗂️ Input Structure

Your `wiki_crop` folder should look like this:

```
wiki_crop/
└── wiki_crop/
    ├── wiki.mat          ← Main metadata file
    ├── 00/               ← Image folders
    ├── 01/
    ├── 02/
    └── ...
```

## 📊 wiki.mat Structure

The `wiki.mat` file contains the following attributes:

| Attribute | Description |
|-----------|-------------|
| `dob` | Date of birth (Matlab serial date number) |
| `photo_taken` | Year when the photo was taken |
| `full_path` | Relative path to image file |
| `gender` | 0 = female, 1 = male, NaN = unknown |
| `name` | Name of the person |
| `face_location` | Face bounding box [x1, y1, x2, y2] |
| `face_score` | Face detection confidence (higher = better) |
| `second_face_score` | Second face confidence (for filtering multiple faces) |

## 🎯 What the Script Does

### 1. **Load wiki.mat**
   - Reads the Matlab file using `scipy.io.loadmat`
   - Extracts all metadata arrays

### 2. **Calculate Ages**
   ```python
   age = photo_taken_year - birth_year
   ```
   - Converts Matlab datenum to Python datetime
   - Calculates age at time of photo

### 3. **Quality Filtering**
   - ✅ Face detection score ≥ 1.0
   - ✅ Second face score < 0.5 (single face only)
   - ✅ Image size ≥ 256x256 pixels
   - ✅ Face size ≥ 100x100 pixels
   - ✅ Age between 0-120 years

### 4. **Age Group Filtering**
   - Extracts images where: **10 ≤ age ≤ 20** → young group
   - Extracts images where: **70 ≤ age ≤ 80** → old group

### 5. **Train/Validation Split**
   - 80% training, 20% validation
   - Minimum 25 images per split per age group
   - Randomized selection

### 6. **Face Cropping & Saving**
   - Crops face using `face_location` bounding box
   - Resizes if larger than 512px (preserving aspect ratio)
   - Saves as high-quality JPEG (quality=95)
   - Filename format: `{group}_{split}_{index:04d}_age{age}.jpg`

## 📁 Output Structure

```
dataset/
├── training/
│   ├── young_10_20/
│   │   ├── young_10_20_train_0000_age15.jpg
│   │   ├── young_10_20_train_0001_age18.jpg
│   │   └── ... (25+ images)
│   └── old_70_80/
│       ├── old_70_80_train_0000_age72.jpg
│       ├── old_70_80_train_0001_age75.jpg
│       └── ... (25+ images)
└── validation/
    ├── young_10_20/
    │   ├── young_10_20_val_0000_age16.jpg
    │   ├── young_10_20_val_0001_age19.jpg
    │   └── ... (25+ images)
    └── old_70_80/
        ├── old_70_80_val_0000_age73.jpg
        ├── old_70_80_val_0001_age76.jpg
        └── ... (25+ images)
```

## 🚀 Usage

### Method 1: Using PowerShell Script (Recommended)

```powershell
# Navigate to project directory
cd "D:\Generative Deep Learning\AgeBooth-master"

# Run the automated script
.\run_prepare_dataset.ps1
```

### Method 2: Manual Steps

```powershell
# 1. Activate virtual environment
.\myenv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements_dataset.txt

# 3. Run the script
python prepare_dataset_from_mat.py
```

## ⚙️ Configuration

You can adjust the following parameters in `prepare_dataset_from_mat.py`:

```python
# Age groups to extract
AGE_GROUPS = {
    "young_10_20": (10, 20),    # Change age ranges here
    "old_70_80": (70, 80),
}

# Minimum images per split
MIN_TRAIN_IMAGES = 25           # Training images per age group
MIN_VAL_IMAGES = 25             # Validation images per age group

# Quality thresholds
MIN_FACE_SCORE = 1.0            # Face detection confidence
MAX_SECOND_FACE_SCORE = 0.5     # Multiple face threshold
MIN_IMAGE_SIZE = 256            # Minimum image dimensions
MIN_FACE_SIZE = 100             # Minimum face dimensions
```

## 📊 Expected Output

```
================================================================================
AgeBooth Dataset Preparation from wiki.mat
================================================================================

Loading wiki.mat from: .\wiki_crop\wiki_crop\wiki.mat
Found 62328 images in wiki.mat

Creating output directory structure...
Output directory: .\dataset

Filtering images by age and quality...
Processing images: 100%|████████████████████| 62328/62328 [02:15<00:00, 459.23it/s]

Filtering Statistics:
----------------------------------------
Total images processed: 62328
Invalid age: 1523
Low face score: 15234
Multiple faces: 8921
Poor quality: 12456
File not found: 234

Images found per age group:
----------------------------------------
young_10_20: 1856 images
old_70_80: 1243 images

Splitting into training and validation sets...
Saving young_10_20 training images: 100%|████████| 1484/1484 [00:45<00:00, 32.87it/s]
Saving young_10_20 validation images: 100%|██████| 372/372 [00:11<00:00, 32.45it/s]
Saving old_70_80 training images: 100%|██████████| 994/994 [00:30<00:00, 32.67it/s]
Saving old_70_80 validation images: 100%|████████| 249/249 [00:07<00:00, 32.34it/s]

================================================================================
Dataset Preparation Complete!
================================================================================

Final Dataset Statistics:
----------------------------------------
Training Set:
  young_10_20: 1484 images
    → dataset\training\young_10_20
  old_70_80: 994 images
    → dataset\training\old_70_80

Validation Set:
  young_10_20: 372 images
    → dataset\validation\young_10_20
  old_70_80: 249 images
    → dataset\validation\old_70_80

================================================================================

✅ Dataset prepared successfully! You can now proceed with training.
```

## 🔧 Troubleshooting

### Error: "wiki.mat not found"
- Check that `wiki_crop/wiki_crop/wiki.mat` exists
- Verify the folder structure matches the expected layout

### Error: "Failed to load wiki.mat"
- Ensure `scipy` is installed: `pip install scipy`
- Check file is not corrupted (re-download if needed)

### Warning: "Not enough images"
- The IMDB-WIKI dataset may have limited images in certain age ranges
- Option 1: Lower `MIN_TRAIN_IMAGES` and `MIN_VAL_IMAGES`
- Option 2: Expand age ranges (e.g., 8-22 instead of 10-20)
- Option 3: Reduce quality thresholds

### Low quality images
- Increase `MIN_FACE_SCORE` threshold
- Decrease `MAX_SECOND_FACE_SCORE` to filter out multiple faces more strictly

## 📝 Mathematical Details

### Matlab Datenum Conversion

Matlab stores dates as serial date numbers (days since January 0, 0000).

```python
# Matlab: datenum('1990-05-15') = 726833
# Python conversion:
python_date = datetime.fromordinal(726833) - timedelta(days=366)
# Result: datetime(1990, 5, 15)
```

### Age Calculation

```
Age = Year_Photo_Taken - Year_Of_Birth
```

Example:
- Born: 1950 (from Matlab datenum)
- Photo taken: 2015
- Age = 2015 - 1950 = 65 years old

### Face Cropping

Face location format: `[x1, y1, x2, y2]`

```python
# Matlab notation (1-indexed):
face = img(y1:y2, x1:x2, :)

# Python/OpenCV notation (0-indexed):
face = img[y1:y2, x1:x2]
```

## 🔄 Next Steps After Dataset Preparation

1. **Verify Dataset**
   ```powershell
   # Check image counts
   Get-ChildItem .\dataset\training\young_10_20 | Measure-Object
   Get-ChildItem .\dataset\training\old_70_80 | Measure-Object
   ```

2. **Update Training Scripts**
   - Point `INSTANCE_DIR` to `./dataset/training/young_10_20`
   - Point `INSTANCE_DIR` to `./dataset/training/old_70_80`

3. **Start Training**
   ```powershell
   # Train young LoRA
   .\train_young_lora.ps1
   
   # Train old LoRA
   .\train_old_lora.ps1
   ```

## 📚 Dependencies

- **scipy**: Read Matlab .mat files
- **pillow**: Image processing
- **numpy**: Numerical operations
- **opencv-python**: Image manipulation and face cropping
- **tqdm**: Progress bars

Install all: `pip install -r requirements_dataset.txt`

## 🎓 Understanding the Code

### Key Functions

1. **`matlab_datenum_to_age()`**
   - Converts Matlab serial dates to ages
   - Handles edge cases and validation

2. **`parse_face_location()`**
   - Extracts face bounding boxes
   - Validates coordinates

3. **`check_image_quality()`**
   - Applies quality filters
   - Ensures face size and image dimensions

4. **`crop_and_save_face()`**
   - Crops face region
   - Resizes if needed
   - Saves as high-quality JPEG

5. **`process_wiki_mat()`**
   - Main orchestration function
   - Loads data, filters, splits, and saves

## 📄 License

This script is part of the AgeBooth project. Use according to the project license.
