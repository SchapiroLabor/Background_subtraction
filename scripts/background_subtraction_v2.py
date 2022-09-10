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

    arg = parser.parse_args()

    # Standardize paths
    arg.root = abspath(arg.root)
    arg.markers = abspath(arg.markers)
    arg.output = abspath(arg.output)

    return arg



def get_info(input):
    # reading information/metadata about processed
    info = tf.TiffFile(input)
    return info
    
def load(input, channel):
    return input[channel].compute()

# add function to WRITE PYRAMID
  
def main(args):

    print(f"Root file = {args.root}")
    print(f"Marker file = {args.markers}")

    # read in markers.csv file
    markers = pd.read_csv(args.markers)

    # required columns for background subtraction are "Filter", "background" and "exposure", everything else is dropped ^^^^
    try:
        markers = markers[["Filter", "background", "exposure"]]
    except:
        raise error("Markers file needs to contain Filter, background and exposure columns")

    # load image as dask array (unloaded unless directly called)
    img = AI.AICSImage(args.root)
    img = img.get_image_dask_data("CYX")
    assert img.shape[0] == len(markers), "Marker file doesn't match image, check channels"



    # add column with starting indices (which match the image channel indices)
    markers['ind'] = range(0, len(markers))

    # initializing sublist - sublist will indicate which background channel to subtract
    sublist = [-1 for i in range(len(markers))]
    for i, Filter in enumerate(markers[markers.background == True].Filter):
        # in the markers.csv file, the Filter needs to be the exact same!!!
        # sublist needs to be able to handle any background index 
        sublist += np.where(
            np.logical_and(markers.Filter == Filter, np.array(markers.background != True)), 
                markers[markers.background == True].iloc[i].ind+1, 0)
    # add sublist to markers - if sublist == -1, no background subtraction, if other, the value matches the index of the background which needs to be subtracted
    markers["sublist"]=sublist

    # main part
    for channel in markers.ind:
        # if sublist has negative values, no background subtraction because it doesn't match any background channel
        if markers.sublist[channel] < 0:
            # no change - applies to background channels as well
            print(f"Channel {channel} processed, no background subtraction")
        else:
            # extracting just the row from the markers_background dataframe which contains the filter denoted by markers.sublist
            temp_back = markers[markers.ind == markers.sublist[channel]]
            # extracting just the row from markers for the channel being processed
            temp_mark = markers[markers.ind==channel]

            # calculate scalar with which the background will be multiplied (Marker_corr = Marker_root - Background * Exposure_marker / Exposure_background)
            scalar = temp_mark.exposure.values / temp_back.exposure.values

            # create temporary dataframe which will store the multiplied background rounded up to nearest integer
            back = img[markers.sublist[channel]]
            back = np.rint(ne.evaluate("back * scalar")).astype(np.uint16) # slowest step of entire script

            # subtract background from processed channel and if the background intensity for a certain pixel was larger than
            # the processed channel, set intensity to 0 (no negative values)
            #img[channel] = img[channel].compute()
            #img[channel] = img[channel] - back
            img[channel] = np.where(img[channel]<back, 0, img[channel]-back) # not optimized, but works

            del back

            print(f"Channel {channel} processed, {temp_mark.Filter.values[0]} background channel subtracted")

    print("Writing OME-TIFF")
    # write ome-tiff from image to specified output, might need to make dim_order changable, as well as different channel names?
    AIW.OmeTiffWriter.save(
        img,
        args.output,
        dim_order="CYX", 
        channel_names=markers.Filter
    )

if __name__ == '__main__':
    # Read in arguments
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")