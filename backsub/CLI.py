import argparse
import pathlib
from argparse import ArgumentParser as AP

#---CLI-BLOCK---#
def get_args():
    # Script description
    description="""Subtracts background - Lunaphore platform"""

    # Add parser
    parser = AP(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # INPUTS
    inputs = parser.add_argument_group(title="Required Input", 
                                       description="Path to required input file")
    
    inputs.add_argument("-r", 
                        "--root", 
                        dest="root", 
                        action="store",
                        type=pathlib.Path,
                        required=True, 
                        help="File path to root image file.")
    
    inputs.add_argument("-m", 
                        "--markers", 
                        dest="markers", 
                        action="store",
                        type=pathlib.Path,
                        required=True, 
                        help="File path to required markers.csv file"
                        )
    
    inputs.add_argument("--pixel-size", 
                        metavar="SIZE", 
                        dest = "pixel_size", 
                        type=float, 
                        default = None, 
                        action = "store",
                        help="Pixel size in microns"
                        )
    
    inputs.add_argument("--tile-size", 
                        dest="tile_size", 
                        required=False, 
                        type=int, 
                        default=1024, 
                        help="Tile size for pyramid generation"
                        )
    
    inputs.add_argument("--chunk-size", 
                        dest="chunksize", 
                        required=False, 
                        type=int, 
                        default=5000, 
                        help="""Chunk size for dask array (e.g for chunksize 1000, 
                        the image will be split into 1000x1000 chunks)"""
                        )
    
    #VERSION CONTROL
    inputs.add_argument("--version", 
                        action="version", 
                        version="v0.4.1"
                        )
    
    #OUTPUTS
    outputs = parser.add_argument_group(title="Output", 
                                        description="Path to output file"
                                        )

    outputs.add_argument("-o", 
                         "--output", 
                         dest="output", 
                         action="store",
                         type=pathlib.Path,
                         required=True, 
                         help="Path to output file"
                         )
    
    outputs.add_argument("-mo", 
                         "--marker-output", 
                         dest="markerout", 
                         action="store",
                         type=pathlib.Path,
                         required=True, 
                         help="Path to output marker file"
                         )

    arg = parser.parse_args()

    return arg
#---END_CLI-BLOCK---#