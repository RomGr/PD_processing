# Light penetration depth 

[![License](https://img.shields.io/pypi/l/penetration_depth.svg?color=green)](https://github.com/RomGr/penetration_depth/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/penetration_depth.svg?color=green)](https://pypi.org/project/penetration_depth)
[![Python Version](https://img.shields.io/pypi/pyversions/penetration_depth.svg?color=green)](https://python.org)
[![CI](https://github.com/RomGr/PD_processing/actions/workflows/ci.yml/badge.svg)](https://github.com/RomGr/PD_processing/actions/workflows/ci.yml)

<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="images/LOGO_HORAO_NEG 1.jpg" alt="Logo" width="250" height="150">
</div>

### Built With 
[![Python][Python.js]][Python-url]

<!-- ABOUT THE PROJECT -->
## Getting Started

### Installation
1. Open the terminal and run the following command:
    ```sh
    pip install git+https://github.com/RomGr/PD_processing.git
    ```
2. You can also upgrade the version of the [processingmm](https://github.com/RomGr/processingMM.git) program by running the following command:
    ```sh
    pip upgrade git+https://github.com/RomGr/processingMM.git
    ```

### Prepare the data
1. The data should be processed with the last version of the processingmm program
2. All the data (different configurations, i.e. '0_overimposition', '45_overimposition', '90_overimposition', ...) should be in the same folder, with the following architecture:
```
└── data
    ├── configuration_1
    ├── configuration_2
    ├── configuration_3
    └── configuration_4
        ├── measurement_1
        ├── measurement_2
        └── measurement_3
```
3. The annotations should be present in the annotation subfolder of each measurement folder, with the following architecture:
```
└── measurement
    ├── polarimetry
    ├── raw_data
    ├── histology
    └── annotation
        ├── ROI_1.tif
        ├── ROI_2.tif
        └── ROI_3.tif
```  
The ROIs should be numbered, as described above (ROI_1, ROI_2, ...). The number after the ROI should corresponds to the one that will be present in the different results files generated afterwards.

### Data processing
1. Download the following [Jupyter notebook](https://github.com/RomGr/PD_processing/blob/main/PD_analysis.ipynb), and run the cells.
2. Different parameters can be defined:
    1. ```wavelengths```: the different wavelengths processed (in nm). The measurements containing data for these wavelengths will be processed.
    2. ```metric```: the metric that will be used to compute descriptive statistic for the ROIs (i.e. 'mean', 'max' or 'median').
    3. ```iq_size```: the size of the IQ used to quantify the dispersion of the azimuth (i.e. if iq_size = 90, we take the difference between quantile 10 and quantile 90).
    4. ```proportion_azimuth_values```: the proportion of azimuth values used when computing the smallest interval containing it (corresponds to the iq_size values, if iq_size = 90, take the smallest interval containing 80% of the values).
    5. ```parameters```: the different parameter names (i.e. 'depolarization', 'retardance', 'azimuth_pr', 'azimuth_iq' and 'azimuth_sd').
        1. ```depolarization```: the depolarization value.
        2. ```retardance```: the retardance value.
        3. ```azimuth_pr```: the smallest interval containing proportion_azimuth_values% of the azimuth values.
        4. ```azimuth_iq```: the difference between the quantile iq_size and 100-iq_size of the azimuth values.
        5. ```azimuth_sd```: the standard deviation of the azimuth values.
3. The results will be saved in the ```results``` folder, with the following architecture:
```
└── results
    ├── configuration_1
    ├── configuration_2
    ├── configuration_3
    └── configuration_4
        ├── wavelength_1
        ├── wavelength_2
        └── wavelength_3
            └── azimuth
                ├── thickness_1
                ├── thickness_2
                └── thickness_3
                    ├── Histogram_ROI_1.png
                    ├── Histogram_ROI_2.png
                    └── Histogram_ROI_3.png
            └── csv
                ├── azimuth_data.csv
                |
                └── depolarization_data.csv
            └── excel
                ├── azimuth_data.xlsx
                |
                └── depolarization_data.xlsx
            └── imgs
                ├── Azimuth of optical axis
                ├── Depolarization
                ├── intensity
                └── Linear retardance
                    ├── measurement_1.png
                    └── measurement_2.png
            └── individual
                ├── thickness_1
                ├── thickness_2
                └── thickness_3
                    ├── Histogram_ROI_1.png
                    ├── Histogram_ROI_2.png
                    └── Histogram_ROI_3.png
            └── raw_data
                ├── azimuth
                ├── depolarization
                └── retardance
                    ├── thickness_1
                    ├── thickness_2
                    └── thickness_3
                        ├── values_ROI_1.pickle
                        ├── values_ROI_2.pickle
                        └── values_ROI_3.pickle
            └── combined_data_thickness.pickle
    └── prism_files
        ├── wavelength_1
        ├── wavelength_2
        └── wavelength_3
            ├── azimuth_iq_prism.xlsx
            ├── azimuth_pr_prism.xlsx
            ├── azimuth_sd_prism.xlsx
            ├── depolarization_prism.xlsx
            └── retardance_prism.xlsx
``` 
4. Generate the figures by copy-pasting the prism files in the ```prism_files``` folder in the Prism documents.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python.js]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/