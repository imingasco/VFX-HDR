# VFX-HDR

## Introduction

## Usage

To generate the HDR image, run the command below:

```
python3 hdr.py -i INPUT_DIR -o OUTPUT_DIR ...
```

Arguments:

* -i, --input []: Specifies path to the directory containing source images, required
* -o, --output: Specifies path to the directory for the output images, required
* --hdr: Specifies the output file name of HDR image, default: result.hdr
* -a, --align: The source images will be aligned before performing HDR algorithm if specified
* -p, --plot: gcurves and radiance maps will be shown if specified
* -l: Specifies the smoothing factor of the gcurve, default: 50
* -s, --shift: Specifies the maximum shift/offset of MTB alignment algorithm, default: 64
