import os

from PIL import Image, ImageDraw

import numpy as np


def wkt2list(objwkts):
    """
    Convert object pixel coordinates from wkt to list of lists.

    :param objwkts (list): list of object wkts
    :returns: polygon coordinates as list of lists.
    """
    import geodaisy.converters as convert
    import ast
    poly = [ast.literal_eval(convert.wkt_to_geojson(wkt))['coordinates'] for wkt in objwkts]
    return poly


def wkt2list1(objwkts):
    """
    Convert object pixel coordinates from wkt to list of lists.

    Heavy duty version utilising shapely & geojson libraries
    :param objwkts (list): list of object wkts
    :returns: polygon coordinates as list of lists.
    """
    import shapely.wkt
    import geojson
    poly = [geojson.Feature(geometry=shapely.wkt.loads(wkt), properties={}).geometry['coordinates'] for wkt in objwkts]
    return poly


def get_object_polygons(anndf, ann, wkt_type="pixel", _type="wkt"):
    """
    Return object polygons & damage status.

    :param ann (str): location of annotation file to parse
    :param anndf (pandas dataframe): annotation dataframe
    :param wkt_type (str): wkt version to be returned (pixel or geo)
    :param _type (str): format of polygons to be returned (wkt or list of lists)
    :return: polygon list & dmg rating list
    """
    assert _type in ["wkt", "poly"]
    fname = os.path.basename(ann)[:-5]
    img_name = fname + ".png"
    ann = anndf.loc[anndf["img_name"] == img_name]
    if wkt_type == "pixel":
        wktlist = ann['pixwkt'].tolist()
    elif wkt_type == "geo":
        wktlist = ann['geowkt'].tolist()
    dmg = ann['dmg_cat'].tolist()
    try:
        polylist = wkt2list(wktlist)
    except:
        polylist = wkt2list1(wktlist)
    if _type == "wkt":
        return wktlist, dmg
    else:
        return polylist, dmg


def generate_segmap(anndf, ann):
    """
    Generate a segmentation map of specified annotation.

    :param ann (str): location of annotation file to parse
    :param anndf (pandas dataframe): annotation dataframe
    :return: seg map (numpy array)
    """
    plist, _ = get_object_polygons(anndf, ann, _type="poly")
    fname = os.path.basename(ann)[:-5]
    img_name = fname + ".png"
    ann = anndf.loc[anndf["img_name"] == img_name]
    width, height = ann['width'].unique()[0], ann['height'].unique()[0]
    bg = np.zeros((width, height))

    for p in plist:
        polygon = [(coord[0], coord[1]) for coord in p[0]]
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask = np.array(img)
        bg += mask
    return bg
