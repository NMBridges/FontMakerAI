import vtracer
import pathlib
import numpy as np
from PIL import Image


def image_to_svg(image : np.ndarray) -> pathlib.Path:
    '''
    Uses vtracer to trace an image into an SVG file.

    Parameters:
    -----------
    image (np.ndarray): the image to trace

    Returns:
    --------
    pathlib.Path: the path to the output SVG file
    '''
    direc = pathlib.Path(__file__).parent.joinpath("tmp_files")
    direc.mkdir()
    input_path = direc.joinpath("in_img.png")
    output_path = direc.joinpath("out_img.svg")
    im = Image.fromarray(image)
    im.save(input_path)
    vtracer.convert_image_to_svg_py(
        input_path.as_posix(),
        output_path.as_posix(),
        colormode='binary',
        # hierarchical='cutout',
        # mode='spline',
        # filter_speckle=10,
        # layer_difference=16,
        corner_threshold=120,
        # length_threshold=10,
        # max_iterations=10,
        # splice_threshold=150,
        # path_precision=8
    )
    return output_path # TODO: fix to return something in memory so the output directory can be removed


def svg_to_cff_tablelist(svg_path : pathlib.Path) -> list:
    '''
    Converts an SVG to CFF format.

    Parameters:
    -----------
    svg_path (pathlib.Path): the path to the SVG file

    Returns:
    --------
    list: the table list
    '''
    raise NotImplementedError