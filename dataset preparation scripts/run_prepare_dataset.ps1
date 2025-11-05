# PowerShell script to prepare dataset from wiki.mat
# Run this after activating your virtual environment

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  AgeBooth Dataset Preparation" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Warning: Virtual environment not detected" -ForegroundColor Yellow
    Write-Host "   Activating myenv..." -ForegroundColor Yellow
    
    if (Test-Path ".\myenv\Scripts\Activate.ps1") {
        & ".\myenv\Scripts\Activate.ps1"
    } else {
        Write-Host "❌ Error: myenv not found. Please create it first:" -ForegroundColor Red
        Write-Host "   python -m venv myenv" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
Write-Host ""

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Cyan
pip install -q -r requirements_dataset.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Failed to install required packages" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Packages installed successfully" -ForegroundColor Green
Write-Host ""

# Check if wiki.mat exists
$wikiMatPath = ".\wiki_crop\wiki_crop\wiki.mat"
if (-not (Test-Path $wikiMatPath)) {
    Write-Host "❌ Error: wiki.mat not found at: $wikiMatPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please ensure the wiki_crop folder structure is correct:" -ForegroundColor Yellow
    Write-Host "  wiki_crop/" -ForegroundColor Yellow
    Write-Host "    └── wiki_crop/" -ForegroundColor Yellow
    Write-Host "        ├── wiki.mat" -ForegroundColor Yellow
    Write-Host "        ├── 00/" -ForegroundColor Yellow
    Write-Host "        ├── 01/" -ForegroundColor Yellow
    Write-Host "        └── ..." -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Found wiki.mat at: $wikiMatPath" -ForegroundColor Green
Write-Host ""

# Run the dataset preparation script
Write-Host "Starting dataset preparation..." -ForegroundColor Cyan
Write-Host "This may take 10-20 minutes depending on your system..." -ForegroundColor Yellow
Write-Host ""

python prepare_dataset_from_mat.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Error: Dataset preparation failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Dataset Preparation Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review the dataset in the 'dataset' folder" -ForegroundColor White
Write-Host "  2. Run training for young LoRA (10-20 years)" -ForegroundColor White
Write-Host "  3. Run training for old LoRA (70-80 years)" -ForegroundColor White
Write-Host ""
