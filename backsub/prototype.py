#standard libraries
import pathlib
import ome_types
import pandas as pd
import numpy as np
import tifffile as tifff
from loguru import logger
from skimage.transform import pyramid_gaussian
import time
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

def is_pyramid(img_path):
    """
    Checks if image has pyramidal levels

    """
    
    with tifff.TiffFile(img_path) as tif:
        levels=len(tif.series[0].levels)
        pyramid=levels > 1
    return pyramid,levels

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


def write_pyramid(img_instances,
                    src_img_path,
                    outdir,
                    levels,
                    file_name,
                    img_data_type,
                    mpp,
                    ):

    outdir.mkdir(parents=True, exist_ok=True)
    out_file_path= outdir / f'{file_name}.tif'
    sublayers=levels-1

    with tifff.TiffWriter(out_file_path, ome=False, bigtiff=True) as tif:
        #write first the original resolution image,i.e. first layer
        for img,pyramid_action,chann_idx in img_instances:

            if pyramid_action=="calculate":
                pyramid_levels=pyramid_gaussian( img, max_layer=sublayers, preserve_range=True,order=1,sigma=1)
            
            elif pyramid_action=="extract":
                pyramid_levels=( tifff.imread(src_img_path,series=0,level=L,key=chann_idx) for L in range(levels)  )#this is a generator


            for n,layer in enumerate(pyramid_levels):
                if n==0:
                    print(layer.shape)
                    tif.write(
                        layer.astype(img_data_type),
                        description="",
                        subifds=sublayers,
                        metadata=False,  # do not write tifffile metadata
                        tile=(256, 256),
                        photometric='minisblack'
                        )
                elif n>0:
                    print(layer.shape)
                    tif.write(
                        layer.astype(img_data_type),
                        subfiletype=1,
                        metadata=False,
                        tile=(256, 256),
                        photometric='minisblack',
                        compression="lzw"#lzw works better when saving channel-by-channel and jpeg 2000 when saving the whole stack at once
                        )
                
    #tifff.tiffcomment(out_file_path, ome_xml)

def subtract_channels(img_path, markers_info,ref_dtype):

    #This function executes the background substraction using generators, each element of the generator
    #is a tuple with 3 values, such that tuple=(img_with_backsub[array],calculate or extract [str],pyramid_from_index[int])

    for _,channel in markers_info.iterrows():
        print(channel.marker_name)
        if channel.processed:
            """
            yield ( np.clip( tifff.imread(img_path,series=0,level=0,key=channel.ind) -
                            (np.float16(channel.factor)*np.float16( tifff.imread( img_path,series=0,level=0,key=int(channel.bg_idx) ) ) ),
                            0,
                            65535
                            ).astype(ref_dtype),
                    "calculate",
                    np.nan
                    )
            """
            yield (tifff.imread(img_path,series=0,level=0,key=channel.ind),
                   "create",
                   np.nan
                   )

            
        else:
            yield ( np.nan,
                   "extract",
                   int(channel.ind)
                   )

            





def main():
    args=CLI.get_args()
    
    in_path = args.root

    out_path = args.output
    # pixel data is read into RAM lazily, cannot overwrite input file
    assert out_path != in_path

    # Detect pixel size in ome-xml
    pixel_size = detect_pixel_size(in_path, args.pixel_size)
    if pixel_size is None: pixel_size = 1

    markers = process_markers(pd.read_csv(args.markers))
    markers_updated=markers.loc[ markers.keep] 
    print(markers_updated)

    img_generator=subtract_channels(in_path,markers_updated,"uint16")


    ispyramid,pyramid_levels=is_pyramid(in_path)
    if not ispyramid:
        pass
        #TODO:RAISE WARNING

    """
    write_pyramid(img_generator,
                    in_path,
                    out_path,
                    pyramid_levels,
                    "test",
                    "uint16",
                    0.17,
                    )
    """






if __name__ == '__main__':
    _version = 'v0.4.1'

    # Run script
    st = time.time()
    main()
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")




