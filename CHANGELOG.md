# schapirolabor/background_subtraction: Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.5.0 - [2025.11.##]

Rework of Backsub to not have Palom as a dependency reducing the environment size and making it lightweight, and reducing the output file size, while keeping the time and memory usage efficiency.

### `Added`
- `compression` parameter
- hidden argument `comet`, which extracts the metadata on-the-fly for Lunaphore COMET images. When using this argument, the `markers.csv` file is not required.
- two RAM profiles: (1) default, uses moderate RAM. (2) Uses approximately half of the default RAM at the cost of a slight loss in precision of the calculation of the downsized dimensions of the pyramidal output image. This means the dimensions of the pyramidal level will differ between profile 1 and 2. The high-resolution level is not affected by this.
- organizes the tool in five scripts: (1) CLI, (2) ome-schema structure, (3) ome-schema writer, (4) background substraction and writing of output image and (5) extraction of metadata from Lunaphore Comet images.
- logger has been re-designed.
- restructured README

### `Fixed`
- output image file-size is reduced by applying lossless compression ("LZW" by default)

### `Removed`
- Palom and OpenCV as dependencies



## v0.4.1 - [2023.11.21]

The script has been rewritten to perform channel subtraction in a RAM-efficient manner - updating is highly recommended. If the output file is much bigger than expected, adjust the `--tile-size` parameter to a smaller value (e.g `512`). Changing the `--chunk-size` parameter may affect performance (lower values increase execution time, higher values increase RAM usage).

### `Added`
- `--chunk-size` parameter for dask array chunking and delayed execution for subtraction that happens while the output pyramidal OME-TIFF is being created.
- Palom's pyramid writer
- `CHANGELOG.md`

### `Fixed`
- Fixed issue with RAM inefficiency - reworked Backsub.

### `Removed`
- `--pyramid` tag introduced in v0.3.4, for smaller images, a smaller tile size should be specified now.

## Versions v0.2.0 and older:

The `markers.csv` file which gives details about the channels needs to contain the following columns: "Filter", "background" and "exposure". An exemplary [markers_old.csv](https://github.com/SchapiroLabor/Background_subtraction/files/9549686/markers.csv) file is given. The "Filter" column should specify the Filter used when acquiring images. If different stains are aquired with the same filter, the *exact same value* needs to be written (including background) as it is used for determining which background channel should be subtracted. The "background" column should contain logical `TRUE` values for channels which represent autofluorescence. The "exposure" column should contain the exposure time used for channel acquisition, and the measure unit should be consistent across the column. Exposure time is used for scaling the value of the background to be comparable to the processed channel. Usage of these versions is strongly disencouraged.