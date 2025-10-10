import ome_schema as schema
import pandas as pd


def create_ome(conformed_markers,info,software_version):
    """
    This function creates an OME-XML file from a pandas dataframe containing the metadata of the tiles.
    Args:
        tile_info (pd.DataFrame): dataframe containing the metadata of the tiles.
        conformed_markers (list): list with the name of the markers in the corresponding order of their appearance in the ome.tif file .
    Returns:
        str: OME-XML file.
    """
    software=f'backsub {software_version}'
    no_of_channels = len(conformed_markers)
    tile_info_dict={
            "tile": no_of_channels *[1],
            "name":conformed_markers ,
            "type": no_of_channels  *[ info["data_type"] ],
            "size_x":no_of_channels *[ info["size_x"] ] ,
           "size_y":no_of_channels *[info["size_y"]],
           "physical_size_x": no_of_channels *[info["pixel_size"]],
           "physical_size_x_unit": no_of_channels *[info["pixel_size_unit"]],
           "physical_size_y": no_of_channels *[info["pixel_size"]],
           "physical_size_y_unit": no_of_channels *[info["pixel_size_unit"]],
           "significant_bits": no_of_channels *["16"]
           }
    tile_info=pd.DataFrame(tile_info_dict)
    
    grouped_tiles = tile_info.groupby(['tile'])
    
    tiles_counter = 0
    image = []
    for tileID, frame in grouped_tiles:
        metadata = schema.INPUTS(frame)
        tiff = schema.TIFF_array(no_of_channels, inputs={'offset': no_of_channels * tiles_counter})
        plane = schema.PLANE_array(no_of_channels, metadata)
        channel = schema.CHANN_array(no_of_channels, metadata)
        pixels = schema.PIXELS_array(channel, plane, tiff, metadata)
        image.append(schema.IMAGE_array (pixels, tiles_counter))
        tiles_counter += 1
    ome, ome_xml = schema.OME_metadata(image,software)

    return ome_xml
