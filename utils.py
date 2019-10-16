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

    Each pixel of the segmentation map will indicate the status of the bldg
    as defined in dmg_dict.

    Note: 'none' indicates its a pre disaster annotation.

    :param ann (str): location of annotation file to parse
    :param anndf (pandas dataframe): annotation dataframe
    :return: seg map (numpy array)
    """
    dmg_dict = {'none': 1,
                'no-damage': 1,
                'minor-damage': 2,
                'major-damage': 3,
                'destroyed': 4,
                'un-classified': 0}

    plist, dmg = get_object_polygons(anndf, ann, _type="poly")
    fname = os.path.basename(ann)[:-5]
    img_name = fname + ".png"
    ann = anndf.loc[anndf["img_name"] == img_name]
    try:
        width, height = ann['width'].unique()[0], ann['height'].unique()[0]
    except:
        width, height = 1024, 1024
    bg = np.zeros((width, height), dtype=np.uint8)

    for p, d in zip(plist, dmg):
        polygon = [(coord[0], coord[1]) for coord in p[0]]
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=dmg_dict[d], fill=dmg_dict[d])
        mask = np.array(img)
        bg = np.maximum(bg, mask)
        # Assert max dmg of bldg
        assert np.max(bg) <= 4
    return bg


def generate_coco(df):
    """
    Generate dataset in MS COCO format.

    :param ann (str): location of annotation file to parse
    :param df (pandas dataframe): pre or post annotation dataframe
    :return: seg map (numpy array)
    """
    categories = [{'id': 1, 'name': 'no-damage'},
                  {'id': 2, 'name': 'minor-damage'},
                  {'id': 3, 'name': 'major-damage'},
                  {'id': 4, 'name': 'destroyed'}]
    images = []
    annotations = []

    # Create Image field
    img_names = list(df.img_name.unique())
    img_ids = [i for i in img_names]
    img_dict = dict(zip(img_names, img_ids))
    width, height = 1024, 1024

    img_field = []

    for im_name, imid in img_dict.items():
        im = {'id': imid, 'file_name': im_name, 'width': width, 'height': height}
        img_field.append(im)

    # Create annotation field

    # Annotation field contains 
    """
    id:
    image_id:
    category_id:
    category_name:
    bbox:
    segmentation:
    area:
    """
    pass