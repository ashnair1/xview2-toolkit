"""Utility scripts for working with xView2 dataset."""

import os
import random
import json

from PIL import Image, ImageDraw

import numpy as np

import geojson
import shapely.wkt
from shapely.geometry import LineString


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
    poly = [geojson.Feature(geometry=shapely.wkt.loads(wkt), properties={}).geometry['coordinates'] for wkt in objwkts]
    return poly


def get_object_polygons(anndf, ann, wkt_type="pixel", _type="wkt"):
    """
    Return object polygons & damage status.

    :param ann: location of annotation file to parse
    :param anndf: annotation dataframe
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


def getboxwh(polygon):
    """
    Get the major and minor axis of a polygon's mrr.

    :param polygon: Shapely polygon
    :return: major_axis, minor_axis
    """
    # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
    mbr_points = list(zip(*polygon.minimum_rotated_rectangle.exterior.coords.xy))

    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [LineString((mbr_points[i], mbr_points[i + 1])).length for i in range(len(mbr_points) - 1)]

    # get major/minor axis measurements
    minor_axis = round(min(mbr_lengths))
    major_axis = round(max(mbr_lengths))

    return major_axis, minor_axis


def generate_segmap(anndf, ann):
    """
    Generate a segmentation map of specified annotation.

    Each pixel of the segmentation map will indicate the status of the bldg
    as defined in dmg_dict.

    Note: 'none' indicates its a pre disaster annotation.

    :param ann: location of annotation file to parse
    :param anndf: annotation dataframe
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


def generate_coco(predf, postdf, filename ='xview2'):
    """
    Generate dataset in MS COCO format.

    :param predf: pre disaster annotation dataframe
    :param postdf: post disaster annotation dataframe
    :param filename: name of annotation file
    :return: None
    """
    info = {"description": "xView2 Building Damage Classification Dataset",
            "url": "https://xview2.org/",
            "version": "1.0",
            "year": 2019,
            "contributor": "Defense Innovation Unit (DIU)",
            "date_created": "2019/10/25"}

    cat_field = [{'id': 1, 'name': 'no-damage'},
                 {'id': 2, 'name': 'minor-damage'},
                 {'id': 3, 'name': 'major-damage'},
                 {'id': 4, 'name': 'destroyed'}]

    # Create Image field
    img_names = list(predf.img_name.unique())
    img_ids = [i for i in range(len(img_names))]
    img_dict = dict(zip(img_names, img_ids))
    width, height = 1024, 1024

    img_field = []

    for im_name, imid in img_dict.items():
        im = {'id': imid, 'file_name': im_name, 'width': width, 'height': height}
        img_field.append(im)

    # Create annotation field
    cat = {'un-classified': 1,
           'no-damage': 1,
           'minor-damage': 2,
           'major-damage': 3,
           'destroyed': 4}

    ann_field = []

    # Recreate annotation dataframe with polygons from pre & dmg class from post.

    prdf = predf.copy()
    podf = postdf.copy()

    # Reset indices
    prdf = prdf.reset_index(drop=True)
    podf = podf.reset_index(drop=True)

    # If merging into postdf
    # podf['pixwkt'] = prdf['pixwkt']
    # podf['geowkt'] = prdf['geowkt']

    # If merging into predf
    prdf['dmg_cat'] = podf['dmg_cat']
    prdf['dmg_cat'] = podf['dmg_cat']

    for index, row in prdf.iterrows():
        ann_id = index
        img_id = img_dict[row['img_name']]

        pixwkt = str(row['pixwkt'])
        poly = shapely.wkt.loads(pixwkt)
        # Area
        area = round(poly.area, 2)

        # Category
        dmg_cat = row['dmg_cat']
        catid = cat[row['dmg_cat']]

        # BBox
        box = poly.minimum_rotated_rectangle
        w, h = getboxwh(poly)
        x, y = list(box.exterior.coords)[0]
        bbox = [round(x, 2), round(y, 2), w, h]

        # Segmentation Polygon
        coord = geojson.Feature(geometry=poly)['geometry']['coordinates'][0]
        seg = [[int(round(item, 2)) for sublist in coord for item in sublist]]

        ann = {'id': ann_id,
               'iscrowd': 0,
               'image_id': img_id,
               'area': area,
               'bbox': bbox,
               'category_id': catid,
               'category_name': dmg_cat,
               'segmentation': seg,
               }

        ann_field.append(ann)

    coco = {'annotations': ann_field,
            'categories': cat_field,
            'images': img_field,
            'info': info}

    with open(filename + ".json", 'w') as w:
        json.dump(coco, w)

    print("COCO Conversion Complete")


def generate_coco_split(predf, postdf, split=0.7):
    """
    Generate dataset in MS COCO format.

    :param predf: pre disaster annotation dataframe
    :param postdf: post disaster annotation dataframe
    :param split: train-val split
    :type split: float
    :return: None
    """
    # Split pre-disaster dataframe
    pre = predf['img_name'].unique()
    pre_train = random.sample(list(pre), int(split * len(pre)))
    pre_val = list(set(pre) - set(pre_train))
    train_pre_df = predf[predf['img_name'].isin(pre_train)]
    val_pre_df = predf[predf['img_name'].isin(pre_val)]
    # Reset indices
    train_pre_df = train_pre_df.reset_index(drop=True)
    val_pre_df = val_pre_df.reset_index(drop=True)
    assert len(predf) == len(train_pre_df) + len(val_pre_df)

    # Split post-disaster dataframe
    post = postdf['img_name'].unique()
    post_train = []
    for i in pre_train:
        nl = i.split('_')
        nn = '_'.join(['post' if x == 'pre' else x for x in nl])
        post_train.append(nn)
    post_val = list(set(post) - set(post_train))
    train_post_df = postdf[postdf['img_name'].isin(post_train)]
    val_post_df = postdf[postdf['img_name'].isin(post_val)]

    # Reset indices
    train_post_df = train_post_df.reset_index(drop=True)
    val_post_df = val_post_df.reset_index(drop=True)
    assert len(postdf) == len(train_post_df) + len(val_post_df)

    # Create MS-COCO train val annotations
    print("Processing train split")
    generate_coco(train_pre_df, train_post_df, "train")
    print("Processing val split")
    generate_coco(val_pre_df, val_post_df, "val")
