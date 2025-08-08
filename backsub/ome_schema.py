#!/usr/bin/python
import ome_types
from ome_types.model import OME,Image,Pixels,TiffData,Channel,Plane
import platform


def INPUTS(frame):
    """
    This function creates a dictionary with the metadata of the tiles.
    Args:
        frame (pd.DataFrame): dataframe containing the metadata of the tiles.
        conformed_markers (list): list of tuples with the name of the markers and their corresponding fluorophore.
    Returns:
        dict: dictionary with the metadata of the tiles.
    """
    inputs=frame.to_dict('list')

    return inputs


def TIFF_array(no_of_channels, inputs={'offset':0}):
    """
    This function creates a list of TIFFData objects.
    Args:
        no_of_channels (int): number of channels.
        inputs (dict): dictionary with the metadata of the tiles.
    Returns:
        list: list of TIFFData objects.
    """
    TIFF = [
        TiffData(
            first_c=ch,
            ifd=n,
            plane_count=1
            )
        for n,ch in enumerate(range(0,no_of_channels), start=inputs['offset'])
        ]

    return TIFF


def PLANE_array(no_of_channels, inputs):
    """
    This function creates a list of Plane objects.
    Args:
        no_of_channels (int): number of channels.
        inputs (dict): dictionary with the metadata of the tiles.
    Returns:
        list: list of Plane objects.
    """

    PLANE = [
        Plane(
            the_c=ch,
            the_t=0,
            the_z=0,
            position_x= inputs['position_x'][ch] if 'position_x' in inputs.keys() else 0,
            position_y= inputs['position_y'][ch] if 'position_y' in inputs.keys() else 0,
            position_z=0,
            position_x_unit= inputs['position_x_unit'][ch] if 'position_x_unit'in inputs.keys() else "pixel" ,
            position_y_unit= inputs['position_y_unit'][ch] if 'position_y_unit'in inputs.keys() else "pixel" 
            )
        for ch in range(0,no_of_channels)
        ]

    return PLANE


def CHANN_array(no_of_channels, inputs):
    """
    This function creates a list of Channel objects.
    Args:
        no_of_channels (int): number of channels.
        inputs (dict): dictionary with the metadata of the tiles.
    Returns:
        list: list of Channel objects.
    """

    CHANN = [
        Channel(
            id=f"Channel:{str(ch)}", # 'Channel:{y}:{x}:{marker_name}'.format(x=ch,y=100+int( inputs['tile'][ch] ) ,marker_name=inputs['marker'][ch] )
            name=inputs["name"][ch],
            color=(255,255,255)
            )
        for ch in range(0,no_of_channels)
        ]

    return CHANN


def PIXELS_array(chann_block, plane_block, tiff_block, inputs):
    """
    This function creates a Pixels object.
    Args:
        chann_block (list): list of Channel objects.
        plane_block (list): list of Plane objects.
        tiff_block (list): list of TIFFData objects.
        inputs (dict): dictionary with the metadata of the tiles.
    Returns:
        Pixels: Pixels object.
    """

    PIXELS = Pixels(
        id=f"Pixels:{inputs['tile'][0]}",
        dimension_order='XYCZT',
        size_c=len(chann_block),
        size_t=1,
        size_x=inputs['size_x'][0],
        size_y=inputs['size_y'][0],
        size_z=1,
        type=inputs['type'][0],#bit_depth
        big_endian=False,
        channels=chann_block,
        interleaved=False,
        physical_size_x=inputs['physical_size_x'][0],
        physical_size_x_unit=inputs['physical_size_x_unit'][0],
        physical_size_y=inputs['physical_size_y'][0],
        physical_size_y_unit=inputs['physical_size_y_unit'][0],
        physical_size_z=1.0,
        planes=plane_block,
        significant_bits=inputs['significant_bits'][0],
        tiff_data_blocks=tiff_block
    )

    return PIXELS


def IMAGE_array(pixels_block, imageID):
    """
    This function creates an Image object.
    Args:
        pixels_block (Pixels): Pixels object.
        imageID (int): identifier of the image.
    Returns:
        Image: Image object.
    """
    
    IMAGE = Image(
            id =f'Image:{imageID}',
            pixels=pixels_block
            )

    return IMAGE


def OME_metadata(image_block,software):
    """
    This function creates an OME object.
    Args:
        image_block (list): list of Image objects.
    Returns:
        OME: OME object.
    """
    ome = OME()
    ome.creator = " ".join([software,
                            ome_types.__name__,
                        ome_types.__version__,
                        '/ python version-',
                        platform.python_version()
                        ]
                        )

    ome.images = image_block
    ome_xml = ome_types.to_xml(ome)

    return ome, ome_xml
