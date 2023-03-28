# VFX-HDR

## Usage

To generate the HDR image and LDR image, run the command below:

```
python3 hdr.py -i INPUT_DIR -o OUTPUT_DIR ...
```

Arguments:

* -i, --input: Specifies path to the directory containing source images, required
* -o, --output: Specifies path to the directory for the output images, required
* --hdr: Specifies the output file name of HDR image, default: result.hdr
* --ldr: Specifies the output file name suffix (excluding extension) of LDR image, default: result
* -a, --align: The source images will be aligned before performing HDR algorithm if specified
* -p, --plot: gcurves and radiance maps will be shown if specified
* -l: Specifies the smoothing factor of the gcurve, default: 50
* -s, --shift: Specifies the maximum shift/offset of MTB alignment algorithm, default: 64
* --scale: Specifies the maximum scale of Reinhard's dodge and burn algorithm, note that the program takes lots of time if this argument is set to larger than 6

## Dependency

This program is tested on Python 3.8.10 and the following libraries:

`numpy 1.24.2`
`PIL 7.0.0`
`cv2 4.7.0`
`matplotlib 3.8.0`
`scipy 1.10.1`

