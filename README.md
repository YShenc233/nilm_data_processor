# NILM Dataset Preprocessing Tool

This tool provides a unified interface for preprocessing NILM (Non-Intrusive Load Monitoring) datasets:
- REDD
- UK-DALE
- Refit

## Features

- Unified preprocessing for all three datasets
- Automatic handling of dataset-specific parameters
- Multiple output modes: combined, separate, or both
- Progress tracking with detailed logs
- Options for normalization and data filtering
- Support for custom houses and appliances selection
- Short file names option for combined data

## Installation

```bash
# Required dependencies
pip install numpy pandas tqdm
```



## Usage

Basic usage:

```
python nilm_data_processor.py --dataset redd --data_dir /path/to/redd/data
```

Processing UK-DALE data with specific houses and appliances:

```
python nilm_data_processor.py --dataset ukdale --data_dir /path/to/ukdale/data --houses 1,2,3 --appliances kettle,fridge
```

Processing Refit data with separate files for each appliance:

```
python nilm_data_processor.py --dataset refit --data_dir /path/to/refit/data --mode separate
```

## Parameters

### Required Parameters

- `--dataset`: Dataset to process (redd, ukdale, or refit)
- `--data_dir`: Directory containing dataset

### Output Configuration

- `--output_dir`: Directory to save processed data (default: 'processed_data')

- ```
  --mode
  ```

  : Output mode (default: 'combined')

  - `combined`: One file for all appliances
  - `separate`: One file per appliance
  - `both`: Create both combined and separate files

- `--compress`: Compress output files using gzip

- `--short_names`: Use shorter filenames for combined mode

- `--skip_existing`: Skip processing if output files already exist

### Data Selection

- ```
  --houses
  ```

  : House indices to process, comma separated (e.g., "1,2,3")

  - If not specified, uses default houses for each dataset:
    - REDD: 1, 2, 3, 4, 5, 6
    - UK-DALE: 1, 2, 3, 4, 5
    - Refit: 2, 3, 5, 16

- ```
  --appliances
  ```

  : Appliances to process, comma separated

  - If not specified, uses default appliances for each dataset:
    - REDD: refrigerator, dishwasher, microwave, washer_dryer
    - UK-DALE: kettle, fridge, washing_machine, microwave, dishwasher, toaster
    - Refit: Washing_Machine, Fridge-Freezer, TV, Kettle, Microwave, Dishwasher

- `--process_all_appliances`: Process all available appliances in dataset

### Data Processing

- ```
  --sampling
  ```

  : Resampling interval (e.g., "6s")

  - Defaults: REDD/UK-DALE: "6s", Refit: "7s"

- ```
  --normalize
  ```

  : Normalization method (default: 'mean')

  - `mean`: Z-score normalization ((data - mean) / std)
  - `minmax`: Min-max normalization ((data - min) / (max - min))
  - `none`: No normalization

- `--validation_size`: Fraction of data to use for validation (default: 0.1)

- `--window_size`: Window size for data processing (default: 480)

- `--window_stride`: Window stride for data processing (default: 120)

### Other Options

- `--verbose`: Enable verbose logging

## Examples

### Processing Multiple Houses with Specific Appliances

```
python nilm_data_processor.py --dataset redd --data_dir /path/to/redd/data --houses 1,2,5 --appliances refrigerator,microwave
```

### Processing All Available Appliances

```
python nilm_data_processor.py --dataset ukdale --data_dir /path/to/ukdale/data --process_all_appliances
```

### Creating Both Combined and Separate Files

```
python nilm_data_processor.py --dataset refit --data_dir /path/to/refit/data --mode both --compress
```

### Using Custom Sampling Rate and Normalization

```
python nilm_data_processor.py --dataset redd --data_dir /path/to/redd/data --sampling 10s --normalize minmax
```

## Preprocessing Steps

The preprocessing workflow involves several key steps:

1. Loading Data
   - Loading raw data from specified houses and appliances
   - Handling dataset-specific file structures and formats
2. Time Alignment
   - Converting timestamps to datetime format
   - Resampling data to consistent intervals
   - Aligning main and appliance-specific data
3. Data Cleaning
   - Removing rows with missing values
   - Filtering out negative or abnormally high values
   - Setting small values (< 5W) to zero
   - Clipping values to appliance-specific limits
4. Status Computation
   - Detecting on/off states based on power thresholds
   - Filtering short on/off events using minimum duration constraints
   - Creating binary status arrays (0=off, 1=on)
5. Normalization
   - Mean normalization (Z-score)
   - Min-max normalization
   - Storing normalization parameters for later use
6. Output Generation
   - Creating combined or separate files based on mode
   - Generating metadata and statistics files
   - Optional compression of output files

## Directory Structure

After processing, the following directory structure is created:

```
Copyprocessed_data/
├── redd/
│   ├── combined/
│   │   ├── redd_processed_[appliances].csv
│   │   ├── redd_processed_[appliances]_status.csv
│   │   ├── redd_processed_[appliances]_stats.csv
│   │   └── redd_processed_[appliances]_metadata.csv
│   └── separate/
│       ├── redd_refrigerator.csv
│       ├── redd_refrigerator_status.csv
│       ├── redd_refrigerator_stats.csv
│       └── ...
├── ukdale/
│   └── ...
└── refit/
    └── ...

logs/
├── redd/
│   └── redd_preprocessing_[timestamp].log
├── ukdale/
│   └── ...
└── refit/
    └── ...
```

## Dataset Requirements

### REDD

- Expected directory structure: house_[id]/channel_[num].dat
- Required files: labels.dat, channel_1.dat, channel_2.dat
- Main power measured on channels 1 and 2

### UK-DALE

- Expected directory structure: house_[id]/channel_[num].dat
- Required files: labels.dat, channel_1.dat
- Main power measured on channel 1

### Refit

- Expected structure: Data/House[id].csv, Labels/House[id].txt
- Main power labeled as 'Aggregate'
- Issues flagged in 'Issues' column (removed during preprocessing)