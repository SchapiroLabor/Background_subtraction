import pathlib
from ome_types import from_tiff
import argparse
import pandas as pd
import numpy as np
import warnings


# CLI
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_img",
        required=True,
        type=pathlib.Path,
        help="absolute path of the input image stack (.tif)",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=pathlib.Path,
        help="absolute path of the directory where the output .csv file will be written",
    )

    parser.add_argument(
        "-fn",
        "--output_file_name",
        required=False,
        type=str,
        default="markers.csv",
        help="name of the csv file",
    )

    parser.add_argument(
        "-rr",
        "--remove_background_channels",
        required=False,
        action="store_true",
        help='If applied, values in the "remove" column will be set to TRUE for any channel used as background. Use with caution as background channel exclusion may affect downstream analysis and autofluroescence measurements can be useful.',
    )

    parser.add_argument(
        "-rf",
        "--registration_filter",
        required=False,
        type=str,
        default="DAPI",
        help="Name of filter used for registration (default: DAPI). Note that all channels where the FluorescenceChannel attribute is this value in the metadata will be considered as reference channels.",
    )

    parser.add_argument(
        "-rd",
        "--remove_dapi",
        required=False,
        action="store_true",
        help='If applied, values in the "remove" column will be set to TRUE for any additional DAPI reference except the first. Manual correction of the output markers file can specify other DAPI channels if needed.',
    )

    parser.add_argument(
        "-krm",
        "--keep_registration_markername",
        required=False,
        type=str,
        default="DAPI",
        help='Name of registration marker that should be kept if "remove_dapi" is set to TRUE. Default is "DAPI". If provided value is not present in the marker_name column, the first instance with the registration filter will be kept.',
    )

    args = parser.parse_args()
    return args


def make_marker_names_unique_list(markers):
    """
    Ensures all marker names in a list are unique.
    If a marker appears multiple times, appends an index suffix (_1, _2, ...).

    Example:
        ["DAPI", "TRITC", "Cy5", "DAPI"] → ["DAPI", "TRITC", "Cy5", "DAPI_1"]

    Args:
        markers (list[str]): List of marker names.

    Returns:
        list[str]: List with unique marker names.
    """
    seen = {}
    unique_markers = []

    for name in markers:
        if name in seen:
            seen[name] += 1
            unique_name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
            unique_name = name
        unique_markers.append(unique_name)

    return unique_markers


def meta_from_file(src_img_path, registration_filter):
    # Fetch metadata object
    ome = from_tiff(src_img_path)
    # Fetch image attributes from ome
    ch_names = make_marker_names_unique_list(
        [element.name for element in ome.images[0].pixels.channels]
    )
    exp_times = [element.exposure_time for element in ome.images[0].pixels.planes]

    filters = [
        element.attributes["FluorescenceChannel"]
        for element in ome.structured_annotations[0].value.any_elements[0].children
    ]
    if registration_filter not in filters:
        raise ValueError(
            f"Registration filter '{registration_filter}' not found in image metadata filters: {list(set(filters))}"
        )

    background = [
        None if registration_filter in element else element for element in filters
    ]

    aux_dict = {
        "channel_number": list(range(1, len(ch_names) + 1)),
        # "cycle_number":cycles,
        "marker_name": ch_names,
        "Filter": filters,
        "background": background,
        "exposure": exp_times,
    }

    df = pd.DataFrame(aux_dict)
    return df


def assign_background(
    df, rmv_ref=False, rmv_back=False, ref_marker="DAPI", registration_filter="DAPI"
):
    # Create column ["backsub_process"] indicating which rows will be processed with backsub
    filters_ = df.Filter.unique().tolist()
    # Strings corresponding to filters/background names are set to False, since the are not processed
    backsub_process = df["marker_name"].replace(filters_, value=False, regex=True)
    # Marker_name corresponding to signal will be set to True for processing
    backsub_process = np.where(backsub_process == False, False, True)
    df.insert(df.shape[1], "backsub_process", backsub_process)

    # Assign the latest mention of the autofluorescence channel to the correspondent row in the background column
    rename_background = []

    # List with row indices of background channels to be removed
    for idx, row in df.iterrows():

        if row.backsub_process:
            previous_channels = reversed(df.iloc[:idx].marker_name.to_list())
            for element in previous_channels:
                # Supposes background name is a subset of the channel/marker name
                if row.background in element:
                    rename_background.append(element)
                    break
        else:
            rename_background.append(None)

    df.drop(columns=["backsub_process"], inplace=True)
    df.background = rename_background

    if rmv_ref or rmv_back:
        remove = len(df) * [""]
        df.insert(df.shape[1], "remove", remove)

        if rmv_back:
            df.loc[
                df["background"].isnull() & (df["Filter"] != registration_filter),
                ["remove"],
            ] = "TRUE"

        if rmv_ref:
            df.loc[
                df["background"].isnull() & (df["Filter"] == registration_filter),
                ["remove"],
            ] = "TRUE"
            if ref_marker not in df.marker_name.values:
                first_ref_marker = df[df.Filter == registration_filter].index[0]
                corr_ref_marker = df.loc[first_ref_marker, "marker_name"]
                warnings.warn(
                    f"Reference marker '{ref_marker}' not found. Using first instance of registration filter '{registration_filter}' - Channel '{corr_ref_marker}' instead."
                )
            else:
                first_ref_marker = df[df.marker_name == ref_marker].index[0]
            df.loc[first_ref_marker, "remove"] = ""

    return df


def main():
    args = get_args()
    img_path = args.input_img
    out_dir = args.output_dir
    file_name = args.output_file_name
    registration_filter = args.registration_filter
    global_ref_marker = args.keep_registration_markername

    df = meta_from_file(img_path, ref_marker_name=registration_filter)
    df_updated = assign_background(
        df, args.remove_background_references, global_ref_marker, registration_filter
    )
    df_updated.to_csv(out_dir / file_name, index=False)


if __name__ == "__main__":
    main()
