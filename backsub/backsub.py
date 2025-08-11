#standard libraries
import pandas as pd
import numpy as np
import tifffile as tifff
from loguru import logger
from skimage.transform import pyramid_gaussian,resize
import time
import dask.array as da
from dask.diagnostics import ProgressBar,ResourceProfiler
import tracemalloc
from dask_image.ndfilters import gaussian_filter
from skimage.util import img_as_float32
from skimage.exposure import rescale_intensity
#local libraries
import CLI
import ome_writer

def pyramid_float32(img_arr,sub_levels,down_factor=2):
    std_dev=np.ceil((down_factor - 1)/2)
    height,width=img_arr.shape
    val_range=(np.min(img_arr), np.max(img_arr))
    ref_dtype=img_arr.dtype.name
    factor_schedule=[ 2**i for i in range(1,sub_levels+1) ]
    dim_schedule= [np.ceil([height/f,width/f]) for f in factor_schedule]
    img_aux=img_arr
    for dims in dim_schedule:
        img_dask=da.from_array(img_aux,chunks="auto")
        result=gaussian_filter(da.block(img_dask), sigma=std_dev, order=0,truncate=3)
        img_aux=resize(img_as_float32(result.compute()), dims, order=1, preserve_range=True, anti_aliasing=False)
        img_aux=rescale_intensity(img_aux,out_range=val_range).astype(ref_dtype)
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


def extract_img_props(img_path,pixel_size=None):

    #Checks if image has pyramidal levels
    with tifff.TiffFile(img_path) as tif:
        pyr_levels=len(tif.series[0].levels)
        is_pyramid=pyr_levels > 1
        data_type=tif.series[0].dtype.name
        height,width=tif.series[0].shape[-2::]
        dask_chunksize=da.from_array(tif.pages[0].asarray(), chunks='auto').chunksize
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
               "size_x":width,
               "size_y":height ,
               "chunksize":dask_chunksize
               }
    
    return img_props

def subtract_channels(src_img_path, markers_info,ref_dtype,ref_chunksize):
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

            signal=da.from_array(tifff.imread(src_img_path,series=0,level=0,key=int(channel.ind)), chunks=ref_chunksize)
            
            background=da.from_array(tifff.imread(src_img_path,series=0,level=0,key=int(channel.bg_idx)), chunks=ref_chunksize)

            subtraction=da.clip( signal-( da.rint(factor*background) ),0,65535).astype(ref_dtype)

            print(f"\n {operation_count} Calculating subtraction of background {channel.background}  from {channel.marker_name} signal:")

            with ResourceProfiler(dt=0.25) as resources:
                with ProgressBar():
                    arr=subtraction.compute()

            print(f"Resources used by dask during subtraction {operation_count}:")
            print(resources.results[0],"([sec],[MB],[% CPU usage])")
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
                    calc_lvls=True,
                    save_ram=False
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
                if save_ram:
                    pyramid=pyramid_float32(first_layer,sub_levels)
                else:
                    pyramid=pyramid_gaussian( first_layer, max_layer=sub_levels, preserve_range=True,order=1,sigma=1)
                    next(pyramid)

            elif pyramid_action=="extract":
                #TODO_dont extract first layer just sublayers
                pyramid=( tifff.imread(src_img_path,series=0,level=L,key=chann_idx) for L in range(1,levels)  )


            #skip first layer
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
                
    return out_file_path


def main(version):
    args=CLI.get_args()
    in_path = args.root
    out_path = args.output
    #Pixel data is read into RAM lazily, cannot overwrite input file
    assert out_path != in_path

    # Extract image properties
    src_props = extract_img_props(in_path, args.pixel_size,)
    # Modify pyramid_levels if required
    if src_props["pyramid"]:
        levels=src_props["levels"]
    else:
        levels=args.pyramid_levels

    #Update markers data_frame to include processing information
    markers = process_markers(pd.read_csv(args.markers))
    markers_updated=markers.loc[ markers.keep]
    logger.info("\nTASKS PREVIEW:\n{}",markers_updated)
    tasks=1
    for _,channel in markers_updated.iterrows():
        if channel.processed: 
            print(f"\n({tasks})Channel {channel.marker_name} ({channel.background}) processed, background subtraction")
            tasks+=1
    #Allocate subtraction operation using generators
    img_generator=subtract_channels(in_path,markers_updated,src_props["data_type"],src_props["chunksize"])

    #Write pyramidal file
    out_file_name=f"backsub_{ (in_path.stem).strip(".ome") }.ome.tif"
    
    logger.info(f"\nTASKS PROGRESS" )

    print(f"\nCommencing writing of pyramidal ome.tif file into {out_path}/{out_file_name}")
    print(f"\nCommencing subtraction tasks\n")

    pyramid_abs_path=write_pyramid(
                    img_generator,
                    in_path,
                    out_path,
                    levels,
                    out_file_name,
                    src_props["data_type"],
                    calc_lvls=(not src_props["pyramid"]),
                    save_ram=args.save_ram
                    )
    
    #Write metadata in OME format into the pyramidal file
    channel_names=markers_updated["marker_name"].tolist()
    ome_xml=ome_writer.create_ome(channel_names,src_props,version)
    tifff.tiffcomment(pyramid_abs_path, ome_xml.encode("utf-8"))

    #Write updated markers.csv
    markers_updated = markers_updated.drop(columns=['keep','ind','processed','factor','bg_idx'])
    markers_updated .to_csv(args.markerout, index=False)

    logger.info(f'\nSCRIPT FINISHED PROCESSING TASKS ')
    print(f'\nPyramidal image with {levels} levels was successfully written ')




if __name__ == '__main__':
    _version = 'v0.5.0'

    # Run script
    tracemalloc.start()
    st = time.time()

    main(_version)

    logger.info(f'\nRESOURCES USED')
    print("Memory peak:",((10**(-9))*tracemalloc.get_traced_memory()[1],"GB"))

    rt = time.time() - st
    tracemalloc.stop()
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")
    




