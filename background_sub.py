import argparse
import pathlib

import ome_types
import palom.pyramid
import palom.reader
import pandas as pd
import numpy as np
import copy
import dask.array as da
import numexpr as ne
import math
import cv2
import tifffile
import tqdm
import zarr
from loguru import logger
from argparse import ArgumentParser as AP
from os.path import abspath
import time
import skimage.transform
### Using the palom pyramidal writer by Yu-An Chen: https://github.com/labsyspharm/palom/blob/main/palom/pyramid.py
### Adapted for this use case - only one image can be processed:

class PyramidSetting:

    def __init__(
        self,
        downscale_factor=2,
        tile_size=1024,
        max_pyramid_img_size=1024
    ):
        self.downscale_factor = downscale_factor
        self.tile_size = tile_size
        self.max_pyramid_img_size = max_pyramid_img_size

    def tile_shapes(self, base_shape):
        shapes = np.array(self.pyramid_shapes(base_shape))
        n_rows_n_cols = np.ceil(shapes / self.tile_size)
        tile_shapes = np.ceil(shapes / n_rows_n_cols / 16) * 16
        return [tuple(map(int, s)) for s in tile_shapes]

    def pyramid_shapes(self, base_shape):
        num_levels = self.num_levels(base_shape)
        factors = self.downscale_factor ** np.arange(num_levels)
        shapes = np.ceil(np.array(base_shape) / factors[:,None])
        return [tuple(map(int, s)) for s in shapes]

    def num_levels(self, base_shape):
        factor = max(base_shape) / self.max_pyramid_img_size
        return math.ceil(math.log(factor, self.downscale_factor)) + 1

def format_channel_names(num_channels_each_mosaic, channel_names):
    '''
    format_channel_names(
        [1, 2, 3, 4, 5], ['x', 'x', ['x'], ['x', 'x']]
    )
    >>> [
        'x_1',
        'x_2',
        'x_3',
        'x_4',
        'x_5',
        'x_6',
        'Mosaic 4_1',
        'Mosaic 4_2',
        'Mosaic 4_3',
        'Mosaic 4_4'
    ]
    '''
    matched_channel_names = []
    for idx, (n, c) in enumerate( zip(channel_names, num_channels_each_mosaic*num_channels_each_mosaic[0]) ):
        c=1
        nl = n
        if type(n) == str:
            nl = [n]
        if len(nl) == 1:
            nl = nl
        matched_channel_names.append(nl)
    return make_unique_str(
        [n for l in matched_channel_names for n in l]
    )

def make_unique_str(str_list):
    if len(set(str_list)) == len(str_list):
        return str_list
    else:
        max_length = max([len(s) for s in str_list])
        str_np = np.array(str_list, dtype=np.dtype(('U', max_length+10)))
        unique, counts = np.unique(str_np, return_counts=True)
        has_duplicate = unique[counts > 1]
        for n in has_duplicate:
            suffixes = [
                f"_{i}"
                for i in range(1, (str_np == n).sum()+1)
            ]
            str_np[str_np == n] = np.char.add(n, suffixes)
    return make_unique_str(list(str_np))

def normalize_mosaics(mosaics):
    dtypes = set(m.dtype for m in mosaics)
    if any([np.issubdtype(d, np.floating) for d in dtypes]):
        max_dtype = np.dtype(np.float32)
    else:
        max_dtype = max(dtypes)
    normalized = []
    for m in mosaics:
        assert m.ndim == 2 or m.ndim == 3
        if m.ndim == 2:
            m = m[np.newaxis, :]
        normalized.append(m.astype(max_dtype, copy=False))
    return normalized


def write_pyramid(
    mosaics,
    output_path,
    pixel_size=1,
    channel_names=None,
    verbose=True,
    downscale_factor=4,
    compression=None,
    is_mask=False,
    tile_size=None,
    save_RAM=False,
    kwargs_tifffile=None
):
    mosaics = normalize_mosaics(mosaics)
    ref_m = mosaics[0]
    path = output_path
    num_channels = count_num_channels(mosaics)
    base_shape = ref_m.shape[1:3]
    assert int(downscale_factor) == downscale_factor
    assert downscale_factor < min(base_shape)
    pyramid_setting = PyramidSetting(
        downscale_factor=int(downscale_factor),
        tile_size=max(ref_m.chunksize)
    )
    num_levels = pyramid_setting.num_levels(base_shape)
    tile_shapes = pyramid_setting.tile_shapes(base_shape)
    shapes = pyramid_setting.pyramid_shapes(base_shape)

    if tile_size is not None:
        assert tile_size % 16 == 0, (
            f"tile_size must be None or multiples of 16, not {tile_size}"
        )
        tile_shapes = [(tile_size, tile_size)] * num_levels

    dtype = ref_m.dtype

    software = f'palom {_version}'
    pixel_size = pixel_size
    metadata = {
        'Creator': software,
        'Pixels': {
            'PhysicalSizeX': pixel_size, 'PhysicalSizeXUnit': '\u00b5m',
            'PhysicalSizeY': pixel_size, 'PhysicalSizeYUnit': '\u00b5m'
        },
    }

    if channel_names is not None:
        num_channels_each_mosaic = [
            count_num_channels([m])
            for m in mosaics
        ]
        names = format_channel_names(num_channels_each_mosaic, channel_names)
        if len(names) == num_channels:
            metadata.update({
                'Channel': {'Name': names},
            })

    logger.info(f"Writing to {path}")
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        kwargs = dict(
            metadata=metadata,
            software=software,
            compression=compression
        )
        if kwargs_tifffile is None:
            kwargs_tifffile = {}
        tif.write(
            data=tile_from_combined_mosaics(
                mosaics, tile_shape=tile_shapes[0], save_RAM=save_RAM
            ),
            shape=(num_channels, *shapes[0]),
            subifds=int(num_levels - 1),
            dtype=dtype,
            tile=tile_shapes[0],
            **{**kwargs, **kwargs_tifffile}
        )
        logger.info('Generating pyramid')
        for level, (shape, tile_shape) in enumerate(
            zip(shapes[1:], tile_shapes[1:])
        ):
            if verbose:
                logger.info(f"    Level {level+1} ({shape[0]} x {shape[1]})")
            tif.write(
                data=tile_from_pyramid(
                    path,
                    num_channels,
                    tile_shape=tile_shape,
                    downscale_factor=downscale_factor,
                    level=level,
                    is_mask=is_mask,
                    save_RAM=save_RAM
                ),
                shape=(num_channels, *shape),
                subfiletype=1,
                dtype=dtype,
                tile=tile_shape,
                **{
                    **dict(compression=compression),
                    **kwargs_tifffile
                }
            )


def count_num_channels(imgs):
    for img in imgs:
        assert img.ndim == 2 or img.ndim == 3
    return sum([
        1 if img.ndim == 2 else img.shape[0]
        for img in imgs
    ])
   

def tile_from_combined_mosaics(mosaics, tile_shape, save_RAM=False):
    num_rows, num_cols = mosaics[0].shape[1:3]
    h, w = tile_shape
    n = len(mosaics)
    for idx, m in enumerate(mosaics):
        for cidx, c in enumerate(m):
            # the performance is heavily degraded without pre-computing the
            # mosaic channel
            with tqdm.dask.TqdmCallback(
                ascii=True,
                desc=(
                    f"Assembling mosaic {idx+1:2}/{n:2} (channel"
                    f" {cidx+1:2}/{m.shape[0]:2})"
                ),
            ):
                c = da_to_zarr(c) if save_RAM else c.compute()
            for y in range(0, num_rows, h):
                for x in range(0, num_cols, w):
                    yield np.array(c[y:y+h, x:x+w])
                    # yield m[y:y+h, x:x+w].copy().compute()
            c = None

def tile_from_pyramid(
    path,
    num_channels,
    tile_shape,
    downscale_factor=2,
    level=0,
    is_mask=False,
    save_RAM=False
):
    # workaround progress bar
    # https://forum.image.sc/t/tifffile-ome-tiff-generation-is-taking-too-much-ram/41865/26
    pbar = tqdm.tqdm(total=num_channels, ascii=True, desc=f'Processing channel')
    for c in range(num_channels):
        img = da.from_zarr(zarr.open(tifffile.imread(
            path, series=0, level=level, aszarr=True
        )))
        if img.ndim == 2:
            img = img.reshape(1, *img.shape)
        img = img[c]
        # read using key seems to generate a RAM spike
        # img = tifffile.imread(path, series=0, level=level, key=c)
        if not is_mask:
            img = img.map_blocks(
                cv2.blur,
                ksize=(downscale_factor, downscale_factor), anchor=(0, 0)
            )
        img = da_to_zarr(img) if save_RAM else img.compute()
        num_rows, num_columns = img.shape
        h, w = tile_shape
        h *= downscale_factor
        w *= downscale_factor
        last_c = range(num_channels)[-1]
        last_y = range(0, num_rows, h)[-1]
        last_x = range(0, num_columns, w)[-1]
        for y in range(0, num_rows, h):
            for x in range(0, num_columns, w):
                if (y == last_y) & (x == last_x):
                    pbar.update(1)
                    if c == last_c:
                        pbar.close()
                yield np.array(img[y:y+h:downscale_factor, x:x+w:downscale_factor])
        # setting img to None seems necessary to prevent RAM spike
        img = None


def da_to_zarr(da_img, zarr_store=None, num_workers=None, out_shape=None, chunks=None):
    if zarr_store is None:
        if out_shape is None:
            out_shape = da_img.shape
        if chunks is None:
            chunks = da_img.chunksize
        zarr_store = zarr.create(
            out_shape,
            chunks=chunks,
            dtype=da_img.dtype,
            overwrite=True
        )
    da_img.to_zarr(zarr_store, compute=False).compute(
        num_workers=num_workers
    )
    return zarr_store


def process_markers(markers):
    markers['ind'] = range(0, len(markers))
    if 'remove' not in markers:
        markers['remove'] = ["False" for i in range(len(markers))]
    else:
        markers['remove'] = markers['remove'] == True

    markers['keep'] = markers['remove'] == False

    markers = markers.drop(columns=['remove'])

    return markers

def detect_pixel_size(img_path,pixel_size=None):
    if pixel_size is None:
        print('Pixel size overwrite not specified')
        try:
            metadata = ome_types.from_tiff(img_path)
            pixel_size = metadata.images[0].pixels.physical_size_x
        except Exception as err:
            print(err)
            print('Pixel size detection using ome-types failed')
            pixel_size = None
    return pixel_size

def scale_background(background_channel, scalar):
    background_channel = np.rint(ne.evaluate("background_channel * scalar"))
    return np.where(background_channel>65535,65535,background_channel.astype(np.uint16))

def subtract(channel_to_process, background):
    return np.where(channel_to_process<background, 0, channel_to_process-background)

def subtract_channels(channel_to_process, background_channel, scalar, chunk_size):
    channel_to_process = channel_to_process.rechunk(chunk_size)
    background_channel = background_channel.rechunk(chunk_size)
    background = da.map_blocks(scale_background, background_channel, scalar)
    output = da.map_blocks(subtract, channel_to_process, background)
    background = None
    return output

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
    inputs.add_argument("--tile-size", dest="tile_size", required=False, type=int, default=1024, help="Tile size for pyramid generation")
    inputs.add_argument("--save-ram", dest="ram", required=False, default=False, help="Save RAM during pyramid generation")
    inputs.add_argument("--version", action="version", version="v0.4.0")
    
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

def main(args):
    _version = 'v0.4.0'
    in_path = args.root

    # Automatically infer the output filename, if not specified
    if args.output is None:
        stem = in_path.stem
        out_path = in_path.parent / f"{stem}.ome.tif"
    else:
        out_path = pathlib.Path(args.output)
    # pixel data is read into RAM lazily, cannot overwrite input file
    assert out_path != in_path

    # Detect pixel size in ome-xml
    pixel_size = detect_pixel_size(in_path, args.pixel_size)
    if pixel_size is None: pixel_size = 1

    markers = process_markers(pd.read_csv(args.markers))

    # Use palom to pyramidize the input image
    readers = [palom.reader.OmePyramidReader(in_path)]
    mosaics = [readers[0].pyramid[0]]
    mosaics_out = copy.copy(mosaics)
    chunk_size = (512,512)

    for channel in range(len(markers)):
        if markers.background.isnull()[channel] == True:
            print(f"Channel {markers.marker_name[channel]} ({channel}) processed, no background subtraction")
        else:
            background_marker = markers.iloc[np.array(markers.marker_name == markers.background[channel])]
            scalar = markers[markers.ind == channel].exposure.values / background_marker.exposure.values
            mosaics_out[0][channel] = subtract_channels(mosaics[0][channel], mosaics[0][background_marker.ind.values[0]], scalar, chunk_size)
            print(f"Channel {markers.marker_name[channel]} ({channel}) processed, background subtraction")

    # removes channels from the image as specified in the markers file
    mosaics_out[0] = mosaics_out[0][np.where(markers.keep)[0]]
    channel_names = list(markers.marker_name[markers.keep])

    write_pyramid(
        mosaics_out, out_path, channel_names=channel_names, downscale_factor=2, pixel_size=pixel_size, save_RAM=False
    )
    markers = markers[markers.keep]
    markers = markers.drop(columns=['keep','ind'])
    markers.to_csv(args.markerout, index=False)

if __name__ == '__main__':
    _version = 'v0.4.0'
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")
