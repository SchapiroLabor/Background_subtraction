#!/usr/bin/env python
# -*- coding: utf-8 -*-
#standard libraries
import pandas as pd
import numpy as np
import tifffile as tifff
from loguru import logger
from skimage.transform import resize,rescale
import time
import dask.array as da
from dask.diagnostics import ProgressBar,ResourceProfiler
import tracemalloc
from dask_image.ndfilters import gaussian_filter
from skimage.util import img_as_float32
from skimage.exposure import rescale_intensity
import ome_types
import zarr
#local libraries
import backsub.CLI as CLI
import backsub.ome_writer as ome_writer
from backsub import __version__

#Decorator function to measure run-time and memory peak of a function
def memocron(func):
    def wrapper(*args,**kwargs):
        tracemalloc.start()
        st = time.time()
        result=func(*args,**kwargs)
        print(f'\nRESOURCES USED by {func.__name__} function ')
        print("Memory peak:",((10**(-9))*tracemalloc.get_traced_memory()[1],"GB"))
        rt = time.time() - st
        tracemalloc.stop()
        print("Runtime:",f"{rt // 60:.0f}m {rt % 60:.0f}s")
        return result
    return wrapper


def pyramidal_levels(img_arr,sub_levels,dims_schedule=None):
    
    if dims_schedule is None:
        down_factor=2
        height,width=img_arr.shape
        factor_schedule=[ down_factor**i for i in range(1,sub_levels+1) ]
        dims_schedule= [np.rint([height/f,width/f]) for f in factor_schedule]
    else:
        down_factor=np.rint( 
                            np.mean( [ dims_schedule[i][0]/dims_schedule[i+1][0] 
                                        for i in range(0,len(dims_schedule)-1)] 
                                    ) 
                            ).astype("int")

    ref_dtype=img_arr.dtype.name
    val_range=(np.min(img_arr), np.max(img_arr))
    std_dev=np.ceil((down_factor - 1)/2)
    img_aux=img_arr
    for dims in dims_schedule:
        img_dask=da.from_array(img_aux,chunks="auto")
        result=gaussian_filter(img_dask, sigma=std_dev, order=0,truncate=down_factor)
        img_aux=resize(img_as_float32(result.compute()), dims, order=1, preserve_range=True, anti_aliasing=False)
        img_aux=np.rint(rescale_intensity(img_aux,out_range=val_range) ).astype(ref_dtype)
        yield img_aux


def pyramid_save_ram(img_arr,sub_levels):
    chunksize=(2048,2048)#This value was found to be optimal, in terms of balanced RAM and runtime, for uint16 data
    down_factor=2
    #TODO:need to optimize speed and match dimensions of levels with those of the src image
    ref_dtype=img_arr.dtype.name
    border_overlap=down_factor-1
    ref_dtype=img_arr.dtype.name
    rescale_args={
            "scale":1/down_factor,
            "order":1,
            "preserve_range":True,
            "anti_aliasing":True
                    }
    img_aux=img_arr
    for _ in range(sub_levels):
        img_chunk=da.from_array(img_aux, chunks=chunksize)
        float_half=img_chunk.astype("float32")
        dim_rescale=da.map_overlap(rescale, float_half, depth=border_overlap, boundary="reflect", **rescale_args)
        img_aux=(da.rint(dim_rescale).astype(ref_dtype)).compute()
        yield img_aux


def process_markers(markers):
    markers['ind'] = range(0, len(markers))
    if 'remove' not in markers:
        markers['remove'] = ["False" for i in range(len(markers))]
    else:
        markers['remove'] = markers['remove'] == True

    markers['keep'] = markers['remove'] == False

    markers = markers.drop(columns=['remove'])

    markers.insert(markers.shape[1], "processed", ~ markers.background.isnull())

    missing_bgs = [
        bg for bg in markers['background'].dropna().unique()
        if bg not in markers['marker_name'].values
    ]
    if missing_bgs:
        raise ValueError(f"Background markers not found in marker_name column: {missing_bgs}")

    if markers['background'].dropna().empty:
        print("\nWarning: No background markers specified in the 'background' column of the markers CSV file. No background subtraction was performed on the output file.\n")

    scaling_factor=np.full(markers.shape[0],np.nan)
    background_idx=np.full(markers.shape[0],np.nan)

    for channel in range(len(markers)):

        if markers.processed[channel]:
            bg_idx = markers.loc[ markers.marker_name == markers.background[channel],"ind"  ].tolist()

            if len(bg_idx)>1:
                print(f"""Warning: Background channel with name {markers.background[channel]} 
                      appears several times in the column "marker_name".  
                      Only the first occurrence will be used for the subtraction.""" )
            bg_idx=bg_idx[0]
            scaling_factor[channel] = markers.exposure[channel] / markers.exposure[bg_idx]
            background_idx[channel] = bg_idx

    markers.insert(markers.shape[1], "factor", scaling_factor)
    markers.insert(markers.shape[1], "bg_idx", background_idx)

    return markers


def extract_img_props(img_path,downscale_factor,tile_size,pixel_size=None):

    #Extract data_type, pyramidal specs, height, width
    with tifff.TiffFile(img_path) as tif:
        pyr_levels=len(tif.series[0].levels)
        is_pyramid=pyr_levels > 1
        data_type=tif.series[0].dtype.name
        height,width=tif.series[0].shape[-2::]
        dask_chunksize=da.from_array(tif.pages[0].asarray(), chunks='auto').chunksize
        if is_pyramid:
            #dimensions of the reduced resolution layers(subresolution_dimensions)
            subres_dims=[tif.series[0].levels[lvl].pages[0].shape
                             for lvl in range(1,pyr_levels)
                             ]
        else:
            subres_dims=None

    if downscale_factor <= 1:
        raise ValueError("downscale_factor must be greater than 1")
    max_dim = max(height,width)
    extracted_levels = int(np.floor(np.log(max_dim / tile_size) / np.log(downscale_factor)) + 1)
        
    #Try to extract pixel size from ome-xml
    if pixel_size is None:
        print('Pixel size not specified in the arguments (-mpp)')
        try:
            metadata = ome_types.from_tiff(img_path)
            pixel_size = metadata.images[0].pixels.physical_size_x
            pixel_size_unit = metadata.images[0].pixels.physical_size_x_unit
        except Exception as err:
            print(err)
            print('Pixel size or pixel size unit detection using ome-types failed')
            pixel_size = 1
            pixel_size_unit="pixel"
    else:
        pixel_size=pixel_size
        pixel_size_unit="µm"

    
    img_props={"pixel_size":pixel_size,
               "pixel_size_unit":pixel_size_unit,
               "data_type":data_type,
               "pyramid":is_pyramid,
               "levels":pyr_levels,
               "extracted_levels": max(extracted_levels, 1),
               "sub_levels_dims":subres_dims,
               "size_x":width,
               "size_y":height ,
               "chunksize":dask_chunksize
               }
    
    return img_props


def subtract_channels(src_img_path,
                      signal_index,
                      background_index,
                      factor,
                      ref_chunksize,
                      ref_dtype,
                      task_no
                      ):
    factor=np.float32(factor) #limiting precision to float32 saves memory
    signal_as_zarr = zarr.open( tifff.imread( src_img_path, aszarr=True, series=0, level=0,key=int(signal_index) ) )
    background_as_zarr =zarr.open( tifff.imread( src_img_path, aszarr=True, series=0, level=0, key=int(background_index) ) )
    signal=da.from_zarr(signal_as_zarr, chunks=ref_chunksize )
    background=da.from_zarr(background_as_zarr, chunks=ref_chunksize)
    if np.issubdtype(ref_dtype, np.integer):
        info = np.iinfo(ref_dtype)
        subtraction=da.clip(signal-factor*background, info.min, info.max).astype(ref_dtype)
    else: # float32/64, no upper clipping
        subtraction=da.clip(signal-factor*background, 0, None).astype(ref_dtype)
    with ResourceProfiler(dt=0.25) as resources:
        with ProgressBar():
            result=subtraction.compute()
    print(f"Resources used by dask during subtraction {task_no}:")
    print(resources.results[0],"([sec],[MB],[% CPU usage])")
    return result

def extract_sublevels_from_tiff(path, ch, levels):
    with tifff.TiffFile(path) as tif:
        for l in range(1, levels):
            yield tif.series[0].levels[l].pages[ch].asarray()

def write_pyramid(src_img_path,
                  tasks_table,
                    out_file,
                    levels,
                    sub_lvls_dims,
                    src_data_type,
                    pyramid_tile_shape,
                    compression,
                    is_src_pyramid=False,
                    save_ram=False
                    ):

    sub_levels=levels-1

    total_operations=tasks_table['processed'].values.sum() #Count True values
    count=1

    if compression == 'none':
        compression=None

    with tifff.TiffWriter(out_file,bigtiff=True) as tif:
        #write first the original resolution image,i.e. first layer
        for _,channel in tasks_table.iterrows():
            if channel.processed:
                operation_count=f"({count}/{total_operations})"
                print(f"\n {operation_count} Calculating subtraction of background {channel.background} from {channel.marker_name} signal:")
                first_layer=subtract_channels(src_img_path, channel.ind, channel.bg_idx, channel.factor, (4096,4096), src_data_type, operation_count)
                pyramid_action="calculate"
                count+=1
            else:
                first_layer=tifff.imread(src_img_path,series=0,level=0,key=int(channel.ind))

                if (save_ram or not is_src_pyramid):
                    pyramid_action="calculate"
                else:
                    pyramid_action="extract"
            
            tif.write(
                first_layer,
                subifds=sub_levels,
                tile=pyramid_tile_shape,
                photometric='minisblack',
                compression=compression
            )
            
            if pyramid_action=="calculate":
                if save_ram:
                    pyramid=pyramid_save_ram(first_layer,sub_levels)
                else:
                    pyramid=pyramidal_levels(first_layer,sub_levels,sub_lvls_dims)

            elif pyramid_action=="extract":
                pyramid=extract_sublevels_from_tiff(src_img_path,int(channel.ind),levels)

            
            for sub_layer in pyramid:
                    tif.write(
                        sub_layer,
                        subfiletype=1,
                        tile=pyramid_tile_shape,
                        photometric='minisblack',
                        compression=compression
                            # lzw works better when saving channel-by-channel
                            # jpeg2000 works better saving the whole stack at once (not done here)
                            # zlib is an alternative that's slower when writing, but gives better compression ratio
                    )
                
    return out_file

@memocron
def main():
    args=CLI.get_args(__version__)
    in_path = args.input
    out_path = args.output

    # 0) Validate input_path is not the same as output_path, pixel data is read into RAM lazily, cannot overwrite input file
    assert out_path != in_path

    if args.tile_size % 16 != 0:
        raise ValueError("tile_size has to be a multiple of 16")
    if args.compression not in ['none','lzw','deflate','zlib']:
        raise ValueError("Compression has to be one of the following: 'none','lzw','deflate','zlib'")

    tile_shape = (args.tile_size, args.tile_size)

    # 1) Extract image properties
    src_props = extract_img_props(in_path,
                                  args.downscale_factor,
                                  args.tile_size,
                                  args.pixel_size)

    # 2) Modify pyramid_levels if required
    if src_props["pyramid"]:
        levels=src_props["levels"]
    else:
        levels=src_props["extracted_levels"]

    # 3) Read/Create markers table and update it to include the information of the processing tasks
    markers = process_markers(pd.read_csv(args.markers))

    markers_updated=markers.loc[ markers.keep ]

    #4) Write updated markers.csv without appended columns. This file contains the markers information of the final image stack
    markers_preview = markers_updated.drop(columns=['keep','ind','processed','factor','bg_idx'])
    markers_preview["channel_number"] = np.arange(1, len(markers_preview)+1)
    markers_preview.to_csv(args.markerout, index=False)

    logger.info("\nTASKS PREVIEW:\n{}",markers_updated)
    tasks=1
    for _,channel in markers_updated.iterrows():
        if channel.processed: 
            print(f"\n(Task_{tasks}): background subtraction, Channel {channel.marker_name} (Background {channel.background})")
            tasks+=1

    #5) Calculate subtractions and write output file
    out_file=args.output

    logger.info(f"\nTASKS PROGRESS" )

    print(f"\nCommencing writing of pyramidal ome.tif file into {out_path}\n")
    print(f"\nCommencing subtraction tasks\n")
    if args.compression=='none':
        compression=None
    else:
        compression=args.compression
    pyramid_abs_path=write_pyramid(
                    in_path,
                    markers_updated,
                    out_file,
                    levels,
                    src_props["sub_levels_dims"],
                    src_props["data_type"],
                    tile_shape,
                    compression,
                    is_src_pyramid=src_props["pyramid"],
                    save_ram=args.save_ram
                    )
    
    #6) Write metadata in OME format into the pyramidal file
    channel_names=markers_updated["marker_name"].tolist()
    ome_xml=ome_writer.create_ome(channel_names,src_props,__version__)
    tifff.tiffcomment(pyramid_abs_path, ome_xml.encode("utf-8"))

    logger.info(f'\nSCRIPT FINISHED PROCESSING TASKS ')
    print(f'\nPyramidal image with {levels} levels was successfully written{'' if compression is None else f' with {compression} compression'}.')

if __name__ == '__main__':
    main()
