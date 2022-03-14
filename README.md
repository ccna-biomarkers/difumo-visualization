# difumo_segmentation

This utility download and segment DiFuMo atlas, so each component is exactly one independent region.

The output layout and naming convention follows [templateflow](https://github.com/templateflow/tpl-MNI152NLin2009cAsym).

# Usage

```
usage: main.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH] [-d DIM] [-r RES]
               [--version]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH
                        Input difumo path (default: "./data/raw")
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Output segmented difumo path (default:
                        "./data/processed")
  -d DIM, --dim DIM     Number of dimensions in the dictionary. Valid
                        resolutions available are {64, 128, 256, 512, 1024},
                        -1 for all (default: -1)
  -r RES, --res RES     The resolution in mm of the atlas to fetch. Valid
                        options available are {2, 3}, -1 for all (default: -1)
  --version             show program's version number and exit

    Documentation at https://github.com/SIMEXP/difumo_segmentation
```
