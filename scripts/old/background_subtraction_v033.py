from __future__ import print_function, division
from multiprocessing.spawn import import_main_path
import sys
import copy
import argparse
import numpy as np
import tifffile
import zarr
import skimage.transform
from aicsimageio import aics_image as AI
import pandas as pd
import numexpr as ne
from ome_types import from_tiff, to_xml
from os.path import abspath
from argparse import ArgumentParser as AP
import time
import dask
# This API is apparently changing in skimage 1.0 but it's not clear to
# me what the replacement will be, if any. We'll explicitly import
# this so it will break loudly if someone tries this with skimage 1.0.
try:
    from skimage.util.dtype import _convert as dtype_convert
except ImportError:
    from skimage.util.dtype import convert as dtype_convert


# arg parser
def get_args():
    # Script description
    description="""Subtracts background - Lunaphore platform"""

    # Add parser
    parser = AP(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Sections
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-r", "--root", dest="root", action="store", required=True, help="File path to root image file.")
    inputs.add_argument("-m", "--markers", dest="markers", action="store", required=True, help="File path to required markers.csv file")
    inputs.add_argument("--pixel-size", metavar="SIZE", dest = "pixel_size", type=float, default = None, action = "store",help="pixel size in microns; default is 1.0")
    
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

def preduce(coords, img_in, img_out):
    print(img_in.dtype)
    (iy1, ix1), (iy2, ix2) = coords
    (oy1, ox1), (oy2, ox2) = np.array(coords) // 2
    tile = skimage.img_as_float32(img_in[iy1:iy2, ix1:ix2])
    tile = skimage.transform.downscale_local_mean(tile, (2, 2))
    tile = dtype_convert(tile, 'uint16')
    #tile = dtype_convert(tile, img_in.dtype)
    img_out[oy1:oy2, ox1:ox2] = tile

def format_shape(shape):
    return "%dx%d" % (shape[1], shape[0])

def process_markers(markers):
    # add column with starting indices (which match the image channel indices)
    # this should be removed soon
    markers['ind'] = range(0, len(markers))

    # if the 'remove' column is not specified, all channels are kept. If it is 
    # present, it is converted to a boolean indicating which channels should be removed
    if 'remove' not in markers:
        markers['remove'] = ["False" for i in range(len(markers))]
    else:
        markers['remove'] = markers['remove'] == True
    # invert the markers.remove column to indicate which columns to keep
    markers['remove'] = markers['remove'] == False

    return markers

def process_metadata(metadata, markers):
    try:
        metadata.screens[0].reagents = [metadata.screens[0].reagents[i] for i in markers[markers.remove == True].ind]
    except:
        pass
    try:
        metadata.structured_annotations = [metadata.structured_annotations[i] for i in markers[markers.remove == True].ind]
    except:
        pass
    # these two are required
    metadata.images[0].pixels.size_c = len(markers[markers.remove == True])
    metadata.images[0].pixels.channels = [metadata.images[0].pixels.channels[i] for i in markers[markers.remove == True].ind]
    try:
        metadata.images[0].pixels.planes = [metadata.images[0].pixels.planes[i] for i in markers[markers.remove == True].ind]
    except:
        pass
    try:
        metadata.images[0].pixels.tiff_data_blocks[0].plane_count = sum(markers.remove == True)
    except:
        pass
    return metadata

# NaN values return True for the statement below in this version of Python. Did not use math.isnan() since the values
# are strings if present
def isNaN(x):
    return x != x

def subtract_channel(image, markers, channel, background_marker, output):
    scalar = markers[markers.ind == channel].exposure.values / background_marker.exposure.values
    
    # create temporary dataframe which will store the multiplied background rounded up to nearest integer
    # [0] at the end needed to get [x, y] shape, and not have [1, x, y]
    back = copy.copy(image[background_marker.ind])[0]
    # subtract background from processed channel and if the background intensity for a certain pixel was larger than
    # the processed channel, set intensity to 0 (no negative values)
    back = np.rint(ne.evaluate("back * scalar"))
    back = np.where(back>65535,65535,back.astype(np.uint16))
    # set the pixel value to 0 if the image channel value is lower than the scaled background channel value
    # otherwise, subtract.
    output[channel] = np.where(image[channel]<back, 0, image[channel]-back)
    back = None
    return output[channel]

def subtract(img, markers, output):
    # iterating over channels
    for channel in range(len(markers)):
        # this array is used to find the appropriate background channel
        # IMPORTANT - all values in the 'marker_name' column should be unique        

        # if no background channel is specified, channel is kept like if should be
        if markers.background.isnull()[channel] == True:
            # no change - applies to background channels as well
            output[channel] = copy.copy(img[channel])
            print(f"Channel {markers.marker_name[channel]} ({channel}) processed, no background subtraction")

        else:
            # this array is used to find the appropriate background channel
            # IMPORTANT - all values in the 'marker_name' column should be unique
            find_background = np.array(markers.marker_name == markers.background[channel])
            # scalar is the coefficient with which the background pixel intensity is scaled
            # it is the result of dividing the marker exposure time with the background exposure time
            # Marker_corr = Marker_root - Background * Exposure_marker / Exposure_background
            background_marker = markers.iloc[find_background]

            output[channel] = subtract_channel(img, markers, channel, background_marker, output)

            print(f"Channel {markers.marker_name[channel]} ({channel}) processed, {markers.background[channel]} background channel subtracted")
    return output

def remove_back(img, markers):
    #with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    # subset the image based on the remove column in the markers file
    img = img[markers.remove.tolist()]
    print()
    print(f'Image shape: {img.shape}')
    return img
    
def subres_tiles(level, level_full_shapes, tile_shapes, outpath, scale):
    print(f"\n processing level {level}")
    assert level >= 1
    num_channels, h, w = level_full_shapes[level]
    tshape = tile_shapes[level] or (h, w)
    tiff = tifffile.TiffFile(outpath)
    zimg = zarr.open(tiff.aszarr(series=0, level=level-1, squeeze=False))
    for c in range(num_channels):
        sys.stdout.write(
            f"\r  processing channel {c + 1}/{num_channels}"
        )
        sys.stdout.flush()
        th = tshape[0] * scale
        tw = tshape[1] * scale
        for y in range(0, zimg.shape[1], th):
            for x in range(0, zimg.shape[2], tw):
                a = zimg[c, y:y+th, x:x+tw, 0]
                a = skimage.transform.downscale_local_mean(
                    a, (scale, scale)
                )
                if np.issubdtype(zimg.dtype, np.integer):
                    a = np.around(a)
                a = a.astype('uint16')
                yield a

def main(args):
    img_raw = AI.AICSImage(args.root)
    img = img_raw.get_image_dask_data("CYX")

    markers_raw = pd.read_csv(args.markers)
    markers = process_markers(copy.copy(markers_raw))

    output = dask.array.empty_like(img)

    output = subtract(img, markers, output)
    output = remove_back(output, markers)

    markers_raw = markers_raw[markers_raw.remove != True]
    markers_raw = markers_raw.drop("remove", axis = 1)
    markers_raw.to_csv(args.markerout, index=False)

    # Processing metadata - highly adapted to Lunaphore outputs
    metadata = img_raw.metadata
    metadata = process_metadata(img_raw.metadata, markers)
    
    if args.pixel_size != None:
        # If specified, the input pixel size is used
        pixel_size = args.pixel_size
    elif img_raw.metadata.images[0].pixels.physical_size_x != None:
        # If pixel size is not specified, the metadata is checked (the script trusts users more than metadata)
        pixel_size = img_raw.metadata.images[0].pixels.physical_size_x
    else:
        # If no pixel size specified anywhere, use default 1.0
        pixel_size = 1.0

    # construct levels
    tile_size = 1024
    scale = 2

    print()
    dtype = output.dtype
    base_shape = output[0].shape
    num_channels = output.shape[0]
    num_levels = (np.ceil(np.log2(max(base_shape) / tile_size)) + 1).astype(int)
    factors = 2 ** np.arange(num_levels)
    shapes = (np.ceil(np.array(base_shape) / factors[:,None])).astype(int)

    print("Pyramid level sizes: ")
    for i, shape in enumerate(shapes):
        print(f"   level {i+1}: {format_shape(shape)}", end="")
        if i == 0:
            print("(original size)", end="")
        print()
    print()
    print(shapes)  
    
    level_full_shapes = []
    for shape in shapes:
        level_full_shapes.append((num_channels, shape[0], shape[1]))
    level_shapes = shapes
    tip_level = np.argmax(np.all(level_shapes < tile_size, axis=1))
    tile_shapes = [
        (tile_size, tile_size) if i <= tip_level else None
        for i in range(len(level_shapes))
    ]

    # write pyramid
    with tifffile.TiffWriter(args.output, ome=True, bigtiff=True) as tiff:
        tiff.write(
            data = output,
            shape = level_full_shapes[0],
            subifds=int(num_levels-1),
            dtype=dtype,
            resolution=(10000 / pixel_size, 10000 / pixel_size, "centimeter"),
            tile=tile_shapes[0]
        )
        for level, (shape, tile_shape) in enumerate(
                zip(level_full_shapes[1:], tile_shapes[1:]), 1
        ):
            tiff.write(
                data = subres_tiles(level, level_full_shapes, tile_shapes, args.output, scale),
                shape=shape,
                subfiletype=1,
                dtype=dtype,
                tile=tile_shape
            )

    # note about metadata: the channels, planes etc were adjusted not to include the removed channels, however
    # the channel ids have stayed the same as before removal. E.g if channels 1 and 2 are removed,
    # the channel ids in the metadata will skip indices 1 and 2 (channel_id:0, channel_id:3, channel_id:4 ...)
    tifffile.tiffcomment(args.output, to_xml(metadata))
    print()

        
    
if __name__ == '__main__':
    # Read in arguments
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")
