# Light penetration depth 

[![License](https://img.shields.io/pypi/l/penetration_depth.svg?color=green)](https://github.com/RomGr/penetration_depth/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/penetration_depth.svg?color=green)](https://pypi.org/project/penetration_depth)
[![Python Version](https://img.shields.io/pypi/pyversions/penetration_depth.svg?color=green)](https://python.org)
[![CI](https://github.com/RomGr/penetration_depth/actions/workflows/ci.yml/badge.svg)](https://github.com/RomGr/PD_processing/actions/workflows/ci.yml)

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
2. All the data (different configurations) should be in the same folder, with the following architecture:
```
└── data
    ├── condition_1
    ├── condition_2
    ├── condition_3
    └── condition_4
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



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python.js]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/