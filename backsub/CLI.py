import argparse
import pathlib
from argparse import ArgumentParser as AP


# ---CLI-BLOCK---#
def get_args(_version):
    # Script description
    description = """
                Subtracts background from an image (signal) 
                acquired with fluorescence microscopy.  
                Subtraction is carried out via the formula (SignalImage-factor*BackgroundImage), 
                where factor is the ratio between exposure times of both images.
                """

    # Add parser
    parser = AP(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # INPUT GROUPS
    inputs = parser.add_argument_group(title="INPUTS")

    inputs.add_argument(
        "-r",  ## deprecation warning can be added later
        "--root",  ## deprecation warning can be added later
        "-in",
        "--input",
        dest="input",
        action="store",
        type=pathlib.Path,
        required=True,
        help="File path to input image file.",
    )

    inputs.add_argument(
        "-m",
        "--markers",
        dest="markers",
        action="store",
        type=pathlib.Path,
        required=True,
        help="""File path to the markers.csv file containing the list of marker names
                        and their respective background channels.
                        """,
    )

    inputs.add_argument(
        "-mpp",
        "--pixel-size",
        metavar="SIZE",
        dest="pixel_size",
        required=False,
        type=float,
        default=None,
        action="store",
        help="pixel size in microns, i.e. microns per pixel(mpp)",
    )

    inputs.add_argument(
        "-ts",
        "--tile-size",
        dest="tile_size",
        required=False,
        type=int,
        default=256,
        help="""Tile size for image pyramid creation. Has to be a multiple of 16.
                        """,
    )

    inputs.add_argument(
        "-dsf",
        "--downscale-factor",
        dest="downscale_factor",
        required=False,
        type=int,
        default=2,
        help="""Downscale factor for the image pyramid.
                        This value will be only used if the input image is NOT pyramidal.
                        If input image is pyramidal, the number of levels in the output image 
                        will be the same as in the input so the downscale factor is not applied.
                        """,
    )

    inputs.add_argument(
        "-sr",
        "--save-ram",
        action="store_true",
        help="""RAM usage is cut in half when using this flag.  
                        Notice that the dimensions of the reduced resolution levels (sub-levels) of 
                        the output pyramidal image will slightly differ whether or not using this argument.                    
                        """,
    )

    inputs.add_argument(
        "-comp",
        "--compression",
        dest="compression",
        required=False,
        type=str,
        default="lzw",
        choices=["lzw", "none", "deflate", "zlib"],
        help="""If set, the output pyramidal image will be compressed using the specified
                        compression method. Set to "none" for no compression. Default is LZW. An alternative is zlib.
                        """,
    )

    # VERSION CONTROL
    inputs.add_argument("-v", "--version", action="version", version=_version)

    # OUTPUTS
    outputs = parser.add_argument_group(title="OUTPUTS")

    outputs.add_argument(
        "-o",
        "--output",
        dest="output",
        action="store",
        type=pathlib.Path,
        required=True,
        help="File path where the output pyramidal OME-TIFF will be saved.",
    )

    outputs.add_argument(
        "-mo",
        "--marker-output",
        dest="markerout",
        action="store",
        type=pathlib.Path,
        required=True,
        help="Path to the output .csv marker file matching the channels in the output image.",
    )

    arg = parser.parse_args()

    return arg


# ---END_CLI-BLOCK---#
