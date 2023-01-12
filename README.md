# Background_subtraction

Background subtraction for COMET platform

## Introduction

If there are background (autofluorescence) channels present in a `.tif` image, background subtraction should be performed so as not to skew the quantification counts of markers. The most precise way of subtracting background would be on a pixel-to-pixel basis. An alternative would be on a cell basis by just subtracting the background measurements from the marker measurements for each cell, however, for visual inspection of images, as well as future use of images as figures in published work, it is preferred to use this.

Background subtraction is performed using the following formula:

Marker<sub>*corrected*</sub> = Marker<sub>*raw*</sub> - Background / Exposure<sub>*Background*</sub> * Exposure<sub>*Marker*</sub>


## Usage 

### Markers file

#### Version v0.3.0:
The `markers.csv` file which gives details about the channels needs to contain the following columns: "marker_name", "background" and "exposure". An exemplary [markers.csv](https://github.com/SchapiroLabor/Background_subtraction/example/markers.csv) file is given. The "marker_name" column should indicate the marker for the acquired channel and all values should be unique. The "background" column should indicate the marker name of the channel which needs to be subtracted. This value must match the "marker_name" value of the background channel. The "exposure" column should contain the exposure time used for channel acquisition, and the measure unit should be consistent across the column. Exposure time is used for scaling the value of the background to be comparable to the processed channel. The "remove" column should contain logical `TRUE` values for channels which should be exluded in the output image.


#### Versions v0.2.0 and older:
The `markers.csv` file which gives details about the channels needs to contain the following columns: "Filter", "background" and "exposure". An exemplary [markers_old.csv](https://github.com/SchapiroLabor/Background_subtraction/files/9549686/markers.csv) file is given. The "Filter" column should specify the Filter used when acquiring images. If different stains are aquired with the same filter, the *exact same value* needs to be written (including background) as it is used for determining which background channel should be subtracted. The "background" column should contain logical `TRUE` values for channels which represent autofluorescence. The "exposure" column should contain the exposure time used for channel acquisition, and the measure unit should be consistent across the column. Exposure time is used for scaling the value of the background to be comparable to the processed channel.


### CLI

The script requires four inputs: 
* the path to the starting image given with `-r` or `--root`
* the path to the output image given with `-o` or `--output`
* the path to the `markers.csv` file given with `-m` or `--markers`
* the path to the markers output file given with `-mo` or `--markerout`


### Output

The output file will be a pyramidal `ome.tif` file containing the processed channels with a subtracted scaled background. The channels tagged for removal will be excluded from the final image.

Metadata:
* channel names - as given in input image
* image name - "Background subtracted image" to easily differentiate between it and the original image
* physical pixel size - as given in input image


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
