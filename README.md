# Background_subtraction

Pixel-by-pixel channel subtraction scaled by exposure times, primarily developed for images produced by the COMET platform and to work within the MCMICRO pipeline. Main usecase is autuofluorescence subtraction for multichannel and multicycle images for visualization of images from tissues with high autofluroescence (FFPE), improved segmentation, and quantification (if the previous two usecases aren't necessary, downstream subtraction of autofluorescent signal is encouraged as the script is memory inefficent).

## Introduction

If there are background (autofluorescence) channels present in a `.tif` image, background subtraction should be performed so as not to skew the quantification counts of markers. The most precise way of subtracting background would be on a pixel-to-pixel basis. An alternative would be on a cell basis by just subtracting the background measurements from the marker measurements for each cell, however, for visual inspection of images, as well as future use of images as figures in published work, it is preferred to use this.

Background subtraction is performed using the following formula:

Marker<sub>*corrected*</sub> = Marker<sub>*raw*</sub> - Background / Exposure<sub>*Background*</sub> * Exposure<sub>*Marker*</sub>


## Usage 

The `markers.csv` file which gives details about the channels needs to contain the following columns: "marker_name", "background" and "exposure". An exemplary [markers.csv](https://github.com/SchapiroLabor/Background_subtraction/blob/main/example/markers.csv) file is given. The "marker_name" column should indicate the marker for the acquired channel and all values should be unique. The "background" column should indicate the marker name of the channel which needs to be subtracted. This value must match the "marker_name" value of the background channel. The "exposure" column should contain the exposure time used for channel acquisition, and the measure unit should be consistent across the column. Exposure time is used for scaling the value of the background to be comparable to the processed channel. The "remove" column should contain logical `TRUE` values for channels which should be exluded in the output image.

### Versions v0.4.1 and newer:
The script has been rewritten to perform channel subtraction in a RAM-efficient manner - updating is highly recommended. If the output file is much bigger than expected, adjust the `--tile-size` parameter to a smaller value (e.g `512`). Changing the `--chunk-size` parameter may affect performance (lower values increase execution time, higher values increase RAM usage).


### Versions v0.2.0 and older:
The `markers.csv` file which gives details about the channels needs to contain the following columns: "Filter", "background" and "exposure". An exemplary [markers_old.csv](https://github.com/SchapiroLabor/Background_subtraction/files/9549686/markers.csv) file is given. The "Filter" column should specify the Filter used when acquiring images. If different stains are aquired with the same filter, the *exact same value* needs to be written (including background) as it is used for determining which background channel should be subtracted. The "background" column should contain logical `TRUE` values for channels which represent autofluorescence. The "exposure" column should contain the exposure time used for channel acquisition, and the measure unit should be consistent across the column. Exposure time is used for scaling the value of the background to be comparable to the processed channel.


### CLI

The script requires four inputs: 
* the path to the starting image given with `-r` or `--root`
* the path to the output image given with `-o` or `--output`
* the path to the `markers.csv` file given with `-m` or `--markers`
* the path to the markers output file given with `-mo` or `--markerout`
Optional inputs:
* `--pixel-size` to specify the pixel size of the input image (default: `1.0`), if not specified, the pixel size will be read from the metadata of the input image.
* `--version` to print version and exit (added in v0.3.4)
* `--pyramid` to create a pyramidal output image (default: `True`) (added in v0.3.4)
* `--tile-size` to specify the tile size for the pyramidal output image (default: `1024`) (added in v0.3.4)


### Output

The output image file will be a pyramidal `ome.tif` file containing the processed channels. The channels tagged for removal will be excluded from the final image.
The output markers file will be a `csv` file containing the following columns: "marker_name", "background", "exposure". The "marker_name" column will contain the marker names of the processed channels. The "background" column will contain the marker names of the channels used for subtraction. The "exposure" column will contain the exposure times of the processed channels. 

### Docker usage

If you want to run the background subtraction directly from a pre-configured container with all the required packages, you can either build the docker container yourself or pull it from the Github container registry.

To build the container run:

```
git clone https://github.com/SchapiroLabor/Background_subtraction.git
docker build -t background_subtraction:latest .
docker run background_subtraction:latest python background_sub.py
```

To pull the container from the Github container registry (ghcr.io):

```
## Login to ghcr.io
docker login ghcr.io

## Pull container
docker pull ghcr.io/schapirolabor/background_subtraction:latest
```
