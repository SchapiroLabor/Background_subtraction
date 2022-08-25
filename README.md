# Background_subtraction
Background subtraction for COMET platform

## Usage

### Markers file

If there are background channels present in a `.tif` image, background subtraction should be performed so as not to skew the quantification counts of markers. This could be done on a segmented cell basis, however for visual inspection, a whole image solution is preferred.

The script only works if there the columns "stain", "background" and "exposure" are present in the `markers.csv` file giving details about the channels. An exemplary `[markers.csv](https://github.com/SchapiroLabor/Background_subtraction/files/9423481/markers.csv)` file is given. The "stain" column should specify the channel used when acquiring images and if the same channel is used, the exact same value needs to be written between all cycles (including background) as it is used for determining which background channel should be subtracted. The "background" column should contain logical `TRUE` values for channels which represent backgrounds. The "exposure" column should contain the exposure time used for channel acquisition, and the measure unit should just be consistent across the column. Exposure time is used for scaling the value of the background to be comparable to the processed channel.


### CLI

The script requires three inputs: 
* the path to the raw image given with `-r` or `--raw`
* the path to the output image given with `-o` or `--output`
* the path to the `markers.csv` file given with `-m` or `--markers`

### Output

The output file will be a non-pyramidal `.tif` file containing the processed channels with a subtracted scaled background. The background channels are not included in the file and if no matching background channel is found for a channel, background subtraction is skipped, and the unedited channel appended.


### Docker usage
