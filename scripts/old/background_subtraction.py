from distutils.log import error
import time
import argparse
from argparse import ArgumentParser as AP
from os.path import abspath
import pandas as pd
import tifffile as tf
import numpy as np

def get_args():
    # Script description
    description="""Subtracts background - Lunaphore platform"""

    # Add parser
    parser = AP(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Sections
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-r", "--raw", dest="raw", action="store", required=True, help="File path to raw image file.")
    inputs.add_argument("-m", "--markers", dest="markers", action="store", required=True, help="File path to required markers.csv file")
    
    outputs = parser.add_argument_group(title="Output", description="Path to output file")
    outputs.add_argument("-o", "--output", dest="output", action="store", required=True, help="Path to output file")

    arg = parser.parse_args()

    # Standardize paths
    arg.raw = abspath(arg.raw)
    arg.markers = abspath(arg.markers)
    arg.output = abspath(arg.output)

    return arg



def get_info(input):
    info = tf.TiffFile(input)
    #if (len(info.series)>1):
    #    sys.exit("Multiple scenes")
    # ^^not good because in pyramidal ome-tiffs we get different levels as different scenes/series
    return info
    


def load(input, channel):
    # returns numpy array of the specified channel from input
    # here so it is easier to change if better way found
    return np.array(tf.imread(input, key = channel))



def write(data, output, channel):
    # only first channel does not have to be appended
    if (channel == 0):
        tf.imwrite(output, data, append=False)
    else:
        tf.imwrite(output, data, append = True)
    # writes the channel(data) to the specified output file
    



def main(args):

    print(f"Raw file = {args.raw}")
    print(f"Marker file = {args.markers}")

    markers = pd.read_csv(args.markers)

    # could just be extracted from markers.csv file
    info = get_info(args.raw)
    ch_number = len(info.pages)
    assert ch_number == len(markers), "Marker file doesn't match image, check channels"

    # required columns for background subtraction are "stain", "background" and "exposure", everything else is dropped
    try:
        markers = markers[["stain", "background", "exposure"]]
    except:
        print("Markers file needs to contain stain, background and exposure columns")

    # subset to the only three important columns
    markers = markers[["stain", "background", "exposure"]]
    # add column with starting indices (which match the image channel indices) - not necessary?
    markers['ind'] = range(0, len(markers))
    # subset background markers
    markers_background = markers[markers.background == True][["ind", "stain", "exposure"]]
    # subset markers which need background subtracted
    markers = markers[markers.background != True][["ind", "stain", "exposure"]]

    # list of stains specified as background
    background_stains = [stain for stain in markers_background.stain]
    # initializing sublist - sublist will indicate which background channel to subtract
    sublist = [-1 for i in range(len(markers))]
    for i, stain in enumerate(markers_background.stain):
        # it should be possible for the regex to catch multiple channels unless one is contained in the beginning of the other(s), if so, the code breaks
        # the i+1 is only there so that each row with a regex match gets shifted by one ot differentiate between unmatched and matched rows
        sublist += np.where(markers.stain.str.match(stain)==True, i+1, 0)
    # add sublist to markers - if sublist == -1, no background subtraction, if sublist == 0, first stain in background_stains should be subtracted
    # if sublist == n, n-1th stain in background stains should be subtracted
    markers["sublist"]=sublist

    # create variables containing the background channels!
    background_dictionary = {}  
    for channel in markers_background.ind:  
        # for each channel containing the background, read in the image as a numpy array 
        # data is saved in the background_dictionary with the name "background_channel"
        background_dictionary["background_"+str(channel)]=np.array(tf.imread(args.raw, key = channel))

    # main part
    for channel in markers.ind:
        # if sublist has negative values, no background subtraction because it doesn't match any background channel
        if list(markers.sublist[[channel]])[0] < 0:
            # load the channel which doesn't have a corresponding background channel and write it to output
            write(load(args.raw, channel=channel), args.output, channel=channel)
            print("Channel "+str(channel)+" written into tif, no background subtraction")
        else:
            # extracting just the row from the markers_background dataframe which contains the stain denoted by markers.sublist
            temp_back = markers_background[markers_background.stain == background_stains[list(markers.sublist[[channel]])[0]]]
            # extracting just the row from markers for the channel being processed
            temp_mark = markers[markers.ind==channel]

            # load the channel being processed
            temp = load(args.raw, channel=channel)

            # calculate scalar with which the background will be multiplied (Marker_corr = Marker_raw - Background * Exposure_marker / Exposure_background)
            scalar = list(temp_mark.exposure)[0] / list(temp_back.exposure)[0]

            # create temporary dataframe which will store the multiplied background rounded up to nearest integer
            temp2 = np.rint(np.multiply(background_dictionary["background_"+str(list(temp_back.ind)[0])], scalar))

            # data has to be converted back to 'uint16' - all outliers will be limited to 2^16 (which happens after multiplication with a scalar)
            temp2 = temp2.astype(np.uint16)
            temp = temp.astype(np.uint16)

            # to avoid any negative values, if the processed channel has a pixel value lower than the scaled background, they are both set to 0
            temp[temp<temp2], temp2[temp<temp2] = 0, 0
            # subtract background from processed channel
            temp = np.subtract(temp, temp2)
            del temp2
            
            # append or write the processed channel to the output tif file
            write(temp, args.output, channel)
            print("Channel "+str(channel)+" written with "+str(list(temp_mark.stain)[0])+" background channel subtracted")
            del temp


#how will I handle metadata???

if __name__ == '__main__':
    # Read in arguments
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")
