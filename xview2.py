import os

import json
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import time

import utils


class XView2():
    def __init__(self, img_dir, lbl_dir):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir

        self.imgs = sorted([os.path.join(img_dir, im) for im in os.listdir(self.img_dir)])
        self.jsons = sorted([os.path.join(lbl_dir, lbl) for lbl in os.listdir(self.lbl_dir) if lbl[-5:] == ".json"])

        # Load annotations into memory
        print('Loading annotations into memory...')
        tic = time.time()
        self.anns = [json.load(open(ann, 'r')) for ann in self.jsons]
        print('Done (t={:0.2f}s)'.format(time.time()- tic))

        # Create annotation dictionary
        self.anndict = dict(zip(self.jsons, self.anns))
        # Create annotation dataframe
        print('Creating annotation dataframe...')
        tic = time.time()
        self.anndf = self.generate_dataframe()
        print('Done (t={:0.2f}s)'.format(time.time()- tic))

    def get_object_polygons(self, ann, _type="wkt"):
        p = self.anndict[ann]
        wktlist = [obj['wkt'] for obj in p['features']['xy']]
        if os.path.basename(ann).split('_')[2] == "post":
            dmg = [obj['properties']['subtype'] for obj in p['features']['xy']]
        else:
            dmg = None
        polylist = utils.wkt2list(wktlist)
        if _type == "wkt":
            return wktlist, dmg
        elif _type == "poly":
            return polylist, dmg

    def get_metadata(self, ann):
        return self.anndict[ann]['metadata']

    def pre_post_split(self):
        # Separate into pre and post
        self.pre_imgs = sorted([os.path.join(self.img_dir, im) for im in os.listdir(self.img_dir) if im.split('_')[2] == "pre"])
        self.pre_lbls = sorted([os.path.join(self.lbl_dir, lbl) for lbl in os.listdir(self.lbl_dir) if lbl.split('_')[2] == "pre"])
        self.post_imgs = sorted([os.path.join(self.img_dir, im) for im in os.listdir(self.img_dir) if im.split('_')[2] == "post"])
        self.post_lbls = sorted([os.path.join(self.lbl_dir, lbl) for lbl in os.listdir(self.lbl_dir) if lbl.split('_')[2] == "post"])
        assert len(self.pre_imgs) == len(self.post_imgs)
        assert len(self.pre_lbls) == len(self.post_lbls)
        # Load annotations into memory
        print('Loading annotations into memory...')
        tic = time.time()
        self.pre_anns = [json.load(open(ann, 'r')) for ann in self.pre_lbls]
        self.post_anns = [json.load(open(ann, 'r')) for ann in self.post_lbls]
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        # Create annotation dictionary
        self.pre_ann = dict(zip(self.pre_lbls, self.pre_anns))
        self.post_ann = dict(zip(self.post_lbls, self.post_anns))

    def generate_dataframe(self):
        ann_list = []
        for fname, ann in self.anndict.items():
            # Get features
            feature_type = []
            uids = []
            pixwkts = []
            geowkts = []
            dmg_cats = []

            for i in ann['features']['xy']:
                feature_type.append(i['properties']['feature_type'])
                uids.append(i['properties']['uid'])
                pixwkts.append(i['wkt'])
                if 'subtype' in list(i['properties'].keys()):
                    dmg_cats.append(i['properties']['subtype'])
                else:
                    dmg_cats.append(None)

            for i in ann['features']['lng_lat']:
                geowkts.append(i['wkt'])

            # Get Metadata
            cols = list(ann['metadata'].keys())
            vals = list(ann['metadata'].values())

            newcols = ['obj_type', 'uid', 'pixwkt', 'geowkt', 'dmg_cat'] + cols
            newvals = [[f, u, pw, gw, dmg] + vals for f, u, pw, gw, dmg in zip(feature_type, uids, pixwkts, geowkts, dmg_cats)]
            df = pd.DataFrame(newvals, columns=newcols)
            ann_list.append(df)
        anndf = pd.concat(ann_list, ignore_index=True)
        return anndf

    def show_anns(self, ann):
        metadata = self.get_metadata(ann)
        img_name = metadata['img_name']
        img_path = os.path.join(self.img_dir, img_name)
        imgfile = plt.imread(img_path)

        # Get polygons
        plist, dmg = self.get_object_polygons(ann, _type="poly")
        print("Number of objects =", len(plist))

        plt.figure(figsize=(15,15))
        ax = plt.gca()
        polygons = []
        colordict = {'no-damage': 'w',
                     'minor-damage': 'darseagreen',
                     'major-damage': 'orange',
                     'destroyed': 'red',
                     'un-classified': 'b'}
        color = []
        if len(plist) != 0:
            # Pre_disaster Images
            if dmg == None:
                for p in plist:
                    c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                    polygons.append(Polygon(p[0]))
                    color.append(c)
                    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
                ax.add_collection(p)
                p = PatchCollection(polygons, edgecolors=color, facecolor='none', linewidths=2)
                ax.add_collection(p)

            # Post_disaster Images
            else:
                for p, d in zip(plist,dmg):
                    c = colordict[d]
                    polygons.append(Polygon(p[0]))
                    color.append(c)
                    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
                ax.add_collection(p)
                p = PatchCollection(polygons, edgecolors=color, facecolor='none', linewidths=2)
                ax.add_collection(p)
        ax.axis('off')
        ax.set_title(os.path.basename(ann[:-5]))
        ax.imshow(imgfile)
        plt.show()


if __name__ == "__main__":
    data_dir = "./data"
    folder = "train"
    img_dir = os.path.join(data_dir, folder, 'images')
    lbl_dir = os.path.join(data_dir, folder, 'labels')
    xview = XView2(img_dir, lbl_dir)
    xview.show_anns('./data/train/labels/palu-tsunami_00000002_post_disaster.json')
