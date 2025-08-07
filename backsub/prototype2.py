#standard libraries
import pathlib
import ome_types
import pandas as pd
import numpy as np
import tifffile as tifff
from loguru import logger
from skimage.transform import pyramid_gaussian
import time
import dask.array as da
from dask.diagnostics import Profiler,ProgressBar,ResourceProfiler, CacheProfiler
#local libraries
import CLI


def process_markers(markers):
    markers['ind'] = range(0, len(markers))
    if 'remove' not in markers:
        markers['remove'] = ["False" for i in range(len(markers))]
    else:
        markers['remove'] = markers['remove'] == True

    markers['keep'] = markers['remove'] == False

    markers = markers.drop(columns=['remove'])

    markers.insert(markers.shape[1], "processed", ~ markers.background.isnull())

    scaling_factor=np.full(markers.shape[0],np.nan)
    background_idx=np.full(markers.shape[0],np.nan)

    for channel in range(len(markers)):

        if markers.processed[channel]:
            bg_idx = markers.loc[ markers.marker_name == markers.background[channel],"ind"  ].tolist()

            if len(bg_idx)>1:
                pass
                #TODO: RAISE WARNING OF REPEATED  BACKGROUND ENTRIES IN MARKER_NAME COLUMN
            else:
                bg_idx=bg_idx[0]
                scaling_factor[channel] = markers.exposure[channel] / markers.exposure[bg_idx]
                background_idx[channel] = bg_idx

    markers.insert(markers.shape[1], "factor", scaling_factor)
    markers.insert(markers.shape[1], "bg_idx", background_idx)

    return markers


def extract_img_props(img_path,pixel_size=None):

    #Checks if image has pyramidal levels
    with tifff.TiffFile(img_path) as tif:
        pyr_levels=len(tif.series[0].levels)
        is_pyramid=pyr_levels > 1
        data_type=tif.series[0].dtype.name

    #Try to extract pixel size from ome-xml
    if pixel_size is None:
        print('Pixel size overwrite not specified')
        try:
            metadata = ome_types.from_tiff(img_path)
            pixel_size = metadata.images[0].pixels.physical_size_x
        except Exception as err:
            print(err)
            print('Pixel size detection using ome-types failed')
            pixel_size = None

    img_props={"pixel_size":pixel_size,
               "data_type":data_type,
               "pyramid":is_pyramid,
               "levels":pyr_levels
               }
    
    return img_props

def subtract_channels(src_img_path, markers_info,ref_dtype):
    """
    This function executes the background substraction using generators, each element of the generator
    is a tuple with 3 values, such that:

    tuple=(img_with_backsub[array],calculate or extract pyramid [str],pyramid_from_index[int])

    The second entry of the tuple indicates in the writing process if the pyramid should be calculated using 
    pyramid_gaussian from scikit image.  If extract, the index given in the third entry will fetch all the pyramid
    levels from the original image stack(src_img_path).
    """
    total_operations=markers_info['processed'].values.sum()#Count True values
    count=1
    for _,channel in markers_info.iterrows():

        if channel.processed:
            operation_count=f"({count}/{total_operations})"
            factor=np.float32(channel.factor)#limiting precision to float32 saves memory

            signal=da.from_array(tifff.imread(src_img_path,series=0,level=0,key=int(channel.ind)), chunks='auto')
            
            background=da.from_array(tifff.imread(src_img_path,series=0,level=0,key=int(channel.bg_idx)), chunks=signal.chunksize)

            subtraction=da.clip( signal-( da.rint(factor*background) ),0,65535).astype(ref_dtype)

            logger.info(f"{operation_count} Calculating subtraction of background {channel.background}  from {channel.marker_name} signal  \n")

            with ResourceProfiler(dt=0.25) as resources:
                with ProgressBar():
                    arr=subtraction.compute()

            logger.info(f"Resources used by dask during subtraction {operation_count} \n")
            print(resources.results[0],"([sec,MB,% CPU usage])")
            count+=1
            yield (arr,"calculate",np.nan)

        else:
            yield ( tifff.imread(src_img_path,series=0,level=0,key=int(channel.ind)),"extract",int(channel.ind))


def write_pyramid(img_instances,
                    src_img_path,
                    outdir,
                    levels,
                    file_name,
                    img_data_type,
                    calc_lvls=True
                    ):

    outdir.mkdir(parents=True, exist_ok=True)
    #out_file_path= outdir / f'{file_name}.tif'
    out_file_path=outdir / file_name
    sub_levels=levels-1

    with tifff.TiffWriter(out_file_path, ome=False, bigtiff=True) as tif:
        #write first the original resolution image,i.e. first layer
        for img,pyramid_action,chann_idx in img_instances:
            first_layer=img
                        #Create pyramidal levels accordingly
            if ( pyramid_action=="calculate" or calc_lvls ): 
                pyramid=pyramid_gaussian( first_layer, max_layer=sub_levels, preserve_range=True,order=1,sigma=1)

            elif pyramid_action=="extract":
                pyramid=( tifff.imread(src_img_path,series=0,level=L,key=chann_idx) for L in range(levels)  )

            next(pyramid)#skip first layer
            #Write first layer of the pyramid,i.e. full size image
            tif.write(
                    first_layer.astype(img_data_type),
                    description="",
                    subifds=sub_levels,
                    metadata=False,  # do not write tifffile metadata
                    tile=(256, 256),
                    photometric='minisblack',
                    compression="lzw"
                        )
            
            for sub_layer in pyramid:
                    tif.write(
                        sub_layer.astype(img_data_type),
                        subfiletype=1,
                        metadata=False,
                        tile=(256, 256),
                        photometric='minisblack',
                        compression="lzw"#lzw works better when saving channel-by-channel and jpeg 2000 when saving the whole stack at once
                        )
                
    #tifff.tiffcomment(out_file_path, ome_xml)







def main():
    args=CLI.get_args()
    in_path = args.root
    out_path = args.output

    #Pixel data is read into RAM lazily, cannot overwrite input file
    assert out_path != in_path

    # Extract image properties
    src_props = extract_img_props(in_path, args.pixel_size)
    # Modify pixel_size and pyramid_levels if required
    if src_props["pixel_size"] is None:
        src_props["pixel_size"] = 1

    if src_props["pyramid"]:
        levels=src_props["levels"]
    else:
        levels=args.pyramid_levels#if not given in the CLI args this will take the default value of 8

    #Update markers data_frame to include processing information
    markers = process_markers(pd.read_csv(args.markers))
    markers_updated=markers.loc[ markers.keep]
    logger.info("Data was processed with the following info:\n{}", markers_updated)

    #Allocate subtraction operation using generators 
    img_generator=subtract_channels(in_path,markers_updated,src_props["data_type"])
    
    #Write pyramidal file
    out_file_name=f"backsub_{in_path.stem}"
    logger.info(f"Commencing writing of pyramidal image into {out_path}/{out_file_name}")
    write_pyramid(img_generator,
                    in_path,
                    out_path,
                    levels,
                    out_file_name,
                    src_props["data_type"],
                    calc_lvls=(not src_props["pyramid"])
                    )
    
    logger.info(f'Pyramidal image with {levels} levels was successfully written ')


    

if __name__ == '__main__':
    _version = 'v0.5.0'

    # Run script
    st = time.time()
    main()
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")




