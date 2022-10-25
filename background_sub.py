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
    return markers

def process_metadata(metadata, markers):
    try:
        metadata.screens[0].reagents = [metadata.screens[0].reagents[i] for i in markers[markers.background != True].ind]
    except:
        pass
    try:
        metadata.structured_annotations = [metadata.structured_annotations[i] for i in markers[markers.background != True].ind]
    except:
        pass
    # these two are required
    metadata.images[0].pixels.size_c = len(markers[markers.background != True])
    metadata.images[0].pixels.channels = [metadata.images[0].pixels.channels[i] for i in markers[markers.background != True].ind]
    try:
        metadata.images[0].pixels.planes = [metadata.images[0].pixels.planes[i] for i in markers[markers.background != True].ind]
    except:
        pass
    return metadata



def subtract(img, markers):
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
    return img

def remove_back(img, markers):
    #with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    img = img[markers.background != True]
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

    img = subtract(img, markers)
    img = remove_back(img, markers)

    markers_raw = markers_raw[markers_raw.background != True]
    markers_raw = markers_raw.drop("background", axis = 1)
    markers_raw.to_csv(args.markerout, index=False)

    # Processing metadata - highly adapted to Lunaphore outputs
    metadata = process_metadata(img_raw.metadata, markers)
    metadata = img_raw.metadata

    
    if args.pixel_size != None:
        # If specified, the inputted pixel size is used
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
    dtype = img.dtype
    base_shape = img[0].shape
    num_channels = img.shape[0]
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

    #level_shapes = np.array(level_shapes)
    tip_level = np.argmax(np.all(level_shapes < tile_size, axis=1))
    tile_shapes = [
        (tile_size, tile_size) if i <= tip_level else None
        for i in range(len(level_shapes))
    ]

    # write pyramid

    with tifffile.TiffWriter(args.output, ome=True, bigtiff=True) as tiff:
        tiff.write(
            data = img,
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
        if metadata.images[0].pixels.planes:
            temp_planes = []
            for i, channel_id in enumerate(range(num_channels)):
                temp_plane = metadata.images[0].pixels.planes[channel_id]
                temp_plane.the_c = i
                temp_planes.append(temp_plane)
            metadata.images[0].pixels.planes = temp_planes
        if metadata.images[0].pixels.tiff_data_blocks and len(
                metadata.images[0].pixels.tiff_data_blocks) > 0:
            metadata.images[0].pixels.tiff_data_blocks[0].plane_count = num_channels
        # Write
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
            
        






