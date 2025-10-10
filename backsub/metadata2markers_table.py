import pathlib
from ome_types import from_tiff
import argparse
import pandas as pd
import numpy as np

#CLI
def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_img',
                        required=True,
                        type=pathlib.Path,
                        help='absolute path of the input image stack (.tif)'
                        )
    
    parser.add_argument('-o',
                        '--output_dir',
                        required=True,
                        type=pathlib.Path,
                        help='absolute path of the directory where the output .csv file will be written'
                        )
    
    parser.add_argument('-fn',
                        '--output_file_name',
                        required=False,
                        type=str,
                        default="markers.csv",
                        help='name of the csv file'
                        )
    
    parser.add_argument('-rr',
                        '--remove_background_references',
                        required=False,
                        action='store_true',
                        help='setup the removal of all reference background channels and all DAPI except the first occurrence.'
                        )
    
    parser.add_argument('-rm',
                        '--registration_marker',
                        required=False,
                        type=str,
                        default="DAPI",
                        help='name of the csv file'
                        )

    
    args=parser.parse_args()
    return args

def meta_from_file(src_img_path,ref_marker_name):
    #Fetch metadata object
    ome=from_tiff(src_img_path)
    #Fetch image attributes from ome
    ch_names  = [ element.name for element in ome.images[0].pixels.channels ]
    exp_times = [ element.exposure_time for element in ome.images[0].pixels.planes ]
    #cycles=[int(element.attributes["CycleID"])+1 for element in ome.structured_annotations[0].value.any_elements[0].children]
    filters= [ element.attributes["FluorescenceChannel"] for element in ome.structured_annotations[0].value.any_elements[0].children ]
    background=[None if ref_marker_name in element 
                else element for element in filters]

    aux_dict={"channel_number":list(range(1,len(ch_names)+1)),
              #"cycle_number":cycles,
              "marker_name":ch_names,
              "Filter":filters,
              "background":background,
              "exposure":exp_times
              }
    
    df=pd.DataFrame(aux_dict)
    return df


def assign_background(df,rmv_ref=False,ref_marker="DAPI"):
    #Create column ["backsub_process"] indicating which rows will be processed with backsub
    filters_=df.Filter.unique().tolist()
    #Strings corresponding to filters/background names are set to False, since the are not processed 
    backsub_process=df["marker_name"].replace(filters_,value=False,regex=True)
    #Marker_name corresponding to signal will be set to True for processing
    backsub_process=np.where(backsub_process==False,False,True)
    df.insert(df.shape[1],"backsub_process",backsub_process)

    #Assign the latest mention of the autofluorescence channel to the correspondent row in the background column
    rename_background=[]
    #List with row indices of background channels to be removed
    
    for idx,row in df.iterrows():

        if row.backsub_process:
            previous_channels=reversed(df.iloc[:idx].marker_name.to_list())
            for element in previous_channels:
                #Supposes background name is a subset of the channel/marker name
                if row.background in element:
                    rename_background.append(element)
                    break
        else: 
            rename_background.append(None)

    df.drop(columns=["backsub_process"],inplace=True)
    df.background=rename_background
    if rmv_ref:
        remove_val=len(df)*[""]
        df.insert(df.shape[1],"remove",remove_val)
        df.loc[ df["background"].isnull() , ["remove"] ]="TRUE"
        first_ref_marker=df[df.marker_name== ref_marker].index[0]
        df.loc[first_ref_marker,"remove"]=""
    return df



def main():
    args=get_args()
    img_path=args.input_img
    out_dir=args.output_dir
    file_name=args.output_file_name
    global_ref_marker=args.registration_marker
    
    df=meta_from_file(img_path,global_ref_marker)
    df_updated=assign_background(df,args.remove_background_references,global_ref_marker)
    df_updated.to_csv( out_dir/file_name ,index=False)


if __name__ == '__main__':
    main()



    

