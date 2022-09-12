from distutils.log import error
import time
import argparse
from argparse import ArgumentParser as AP
from os.path import abspath
import pandas as pd
import tifffile as tf
import numpy as np
import numexpr as ne
from aicsimageio import aics_image as AI
from aicsimageio import writers as AIW
from itertools import compress
import ome_types



def get_args():
    # Script description
    description="""Subtracts background - Lunaphore platform"""

    # Add parser
    parser = AP(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Sections
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-r", "--root", dest="root", action="store", required=True, help="File path to root image file.")
    inputs.add_argument("-m", "--markers", dest="markers", action="store", required=True, help="File path to required markers.csv file")
    
    outputs = parser.add_argument_group(title="Output", description="Path to output file")
    outputs.add_argument("-o", "--output", dest="output", action="store", required=True, help="Path to output file")
    outputs.add_argument("-mo", "--marker-output", dest="markerout", action="store", required=True, help="Path to output marker file")

    arg = parser.parse_args()

    # Standardize paths
    arg.root = abspath(arg.root)
    arg.markers = abspath(arg.markers)
    arg.output = abspath(arg.output)
    arg.markerout = abspath(arg.markerout)

    return arg

# add function to WRITE PYRAMID
  
def main(args):

    print(f"Root file = {args.root}")
    print(f"Marker file = {args.markers}")

    # read in markers.csv file
    markers_raw = pd.read_csv(args.markers)

    # required columns for background subtraction are "Filter", "background" and "exposure", everything else is dropped ^^^^
    try:
        markers = markers_raw[["Filter", "background", "exposure"]]
    except:
        raise error("Markers file needs to contain Filter, background and exposure columns")

    # load image as dask array (unloaded unless directly called)
    img_raw = AI.AICSImage(args.root)
    img = img_raw.get_image_dask_data("CYX") # hardcoding shape, channel is first
    assert img.shape[0] == len(markers), "Marker file doesn't match image, check channels"

    # add column with starting indices (which match the image channel indices)
    markers['ind'] = range(0, len(markers))

    # initializing sublist - sublist will indicate which background channel to subtract 
    sublist = [-1 for i in range(len(markers))]
    for i, Filter in enumerate(markers[markers.background == True].Filter):
        # in the markers.csv file, the Filter needs to be the exact same!!!
        # sublist needs to be able to handle any background index 
        # - the index of the background column is put to sublist where 
        # the Filter matches the background
        sublist += np.where(
            np.logical_and(markers.Filter == Filter, np.array(markers.background != True)), 
                markers[markers.background == True].iloc[i].ind + 1, 0)
    # add sublist to markers - if sublist == -1, no background subtraction
    # if other, the value matches the index of the background which needs to be subtracted
    markers["sublist"] = sublist

    # iterating over channels
    for channel in markers.ind:
        # if sublist has negative values, no background subtraction (either because it does
        # not match background, or because it is background)
        if markers.sublist[channel] < 0:
            # no change - applies to background channels as well
            print(f"Channel {channel} processed, no background subtraction")

        else:
            # scalar is the coefficient with which the background pixel intensity is scaled
            # it is the result of dividing the marker exposure time with the background exposure time
            # Marker_corr = Marker_root - Background * Exposure_marker / Exposure_background
            scalar = markers[markers.ind==channel].exposure.values / markers[markers.ind == markers.sublist[channel]].exposure.values
            
            # create temporary dataframe which will store the multiplied background rounded up to nearest integer
            back = img[markers.sublist[channel]]
            back = np.rint(ne.evaluate("back * scalar")).astype(np.uint16) # slowest step of entire script

            # subtract background from processed channel and if the background intensity for a certain pixel was larger than
            # the processed channel, set intensity to 0 (no negative values)
            img[channel] = np.where(img[channel]<back, 0, img[channel]-back) # not optimized, but works
            del back

            print(f"Channel {channel} processed, {markers[markers.ind==channel].Filter.values[0]} background channel subtracted")

    print("Writing OME-TIFF")
    # write ome-tiff from image to specified output, might need to make dim_order changable
    # exlude channels with background set to True (final result omits these channels)
    img = img[markers.background != True]

    AIW.OmeTiffWriter.save(
        img,
        args.output,
        dim_order="CYX", 
        #channel_names=list(markers.Filter[markers.background != True]), useful when testing and bad metadata present
        image_name = "Background subtracted image",
        channel_names = list(compress(img_raw.channel_names, markers.background != True)),
        physical_pixel_sizes = img_raw.physical_pixel_sizes
    )
    # write new markers.csv file and exclude background channels (and the background column)
    markers_raw = markers_raw[markers_raw.background != True]
    markers_raw.drop("background", axis = 1, inplace = True)
    markers_raw.to_csv(args.markerout)


if __name__ == '__main__':
    # Read in arguments
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")