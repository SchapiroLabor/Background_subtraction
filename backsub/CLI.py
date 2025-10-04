import argparse
import pathlib
from argparse import ArgumentParser as AP

#---CLI-BLOCK---#
def get_args():
    # Script description
    description="""
                Subtracts background from an image (signal) 
                acquired with fluorescence microscopy.  
                Subtraction is carried out via the formula (SignalImage-factor*BackgroundImage), 
                where factor is the ratio between exposure times of both images.
                """

    # Add parser
    parser = AP(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # INPUT GROUPS
    inputs = parser.add_argument_group(title="INPUTS")
    input_markers_table = parser.add_mutually_exclusive_group(required=True)
    
    inputs.add_argument("-r", 
                        "--root", 
                        dest="root", 
                        action="store",
                        type=pathlib.Path,
                        required=True, 
                        help="File path to root image file.")
    
    input_markers_table.add_argument("-m", 
                        "--markers", 
                        dest="markers", 
                        action="store",
                        type=pathlib.Path,
                        help="""File path to the markers.csv file containing the list of marker names
                        and its respective background channels.
                        """
                        )
    
    input_markers_table.add_argument("-comet",
                        "--comet_metadata",
                        action='store_true',
                        help=argparse.SUPPRESS
                        )
    
    inputs.add_argument("-mpp",
                        "--pixel-size", 
                        metavar="SIZE", 
                        dest = "pixel_size",
                        required=False,
                        type=float, 
                        default = None, 
                        action = "store",
                        help="pixel size in microns,i.e. microns per pixel(mpp)"
                        )
    
    inputs.add_argument("-pl",
                        "--pyramid_levels", 
                        dest="pyramid_levels", 
                        required=False, 
                        type=int, 
                        default=8, 
                        help="""Total number of pyramid levels.
                        This value will be only used if the input image is NOT pyramidal.
                        If input image is pyramidal, the number of levels in the output image 
                        will be the same as in the input.
                        """
                        )
    
    inputs.add_argument('-sr',
                    '--save_ram',
                    action='store_true',
                    help="""RAM usage is cut in half when using this flag.  
                    Notice that the dimensions of the reduced resolution levels (sub-levels) of 
                    the output pyramidal image will slightly differ when using and not using this argument.                    
                    """
                        )
    #VERSION CONTROL
    inputs.add_argument("--version", 
                        action="version", 
                        version="v0.5.0"
                        )
    
    #OUTPUTS
    outputs = parser.add_argument_group(title="OUTPUTS")

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
                         help="Path to the output .csv marker file"
                         )

    arg = parser.parse_args()

    return arg
#---END_CLI-BLOCK---#
