import json
import os
import random
import shutil
import time

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

import pandas as pd

from PIL import Image

import shapely.wkt

from tqdm import tqdm

import utils


class XView2:
    def __init__(self, img_dir, lbl_dir):
        """
        Constructor of xView helper class for reading, parsing & visualising annotations.

        :param image_folder (str): location to the folder that hosts images.
        :param annotation_file (str): location to the folder that hosts annotations.
        :return:
        """
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir

        self.imgs = sorted([os.path.join(img_dir, im) for im in os.listdir(self.img_dir)])
        self.jsons = sorted([os.path.join(lbl_dir, lbl) for lbl in os.listdir(self.lbl_dir) if lbl[-5:] == ".json"])

        # Load annotations into memory
        print('Loading annotations into memory...')
        tic = time.time()
        self.anns = [json.load(open(ann, 'r')) for ann in self.jsons]
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        # Create annotation dictionary
        self.anndict = dict(zip(self.jsons, self.anns))
        # Create annotation dataframe
        print('Creating annotation dataframe...')
        tic = time.time()
        self.anndf = self.generate_dataframe()
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        _, _ = self.pre_post_split()

        self.colordict = {'none': 'c',
                          'no-damage': 'w',
                          'minor-damage': 'darkseagreen',
                          'major-damage': 'orange',
                          'destroyed': 'red',
                          'un-classified': 'b'}

    def generate_dataframe(self):
        """
        Generate main annotation dataframe.

        :return: anndf (pandas DataFrame)
        """
        # Remove text file
        skipfile = "./skipped.txt"
        if os.path.exists(skipfile):
            os.remove(skipfile)

        ann_list = []
        for k, ann in self.anndict.items():
            # Get features
            feature_type = []
            uids = []
            pixwkts = []
            geowkts = []
            dmg_cats = []
            imids = []
            types = []

            if ann['features']['xy']:
                for i in ann['features']['xy']:
                    feature_type.append(i['properties']['feature_type'])
                    uids.append(i['properties']['uid'])
                    pixwkts.append(i['wkt'])
                    if 'subtype' in list(i['properties'].keys()):
                        dmg_cats.append(i['properties']['subtype'])
                    else:
                        dmg_cats.append("none")
                    imids.append(ann['metadata']['img_name'].split('_')[1])
                    types.append(ann['metadata']['img_name'].split('_')[2])

                for i in ann['features']['lng_lat']:
                    geowkts.append(i['wkt'])

                # Get Metadata
                cols = list(ann['metadata'].keys())
                vals = list(ann['metadata'].values())

                newcols = ['obj_type', 'img_id', 'type', 'pixwkt', 'geowkt', 'dmg_cat', 'uid'] + cols
                newvals = [[f, _id, t, pw, gw, dmg, u] + vals for f, _id, t, pw, gw, dmg, u in
                           zip(feature_type, imids, types, pixwkts, geowkts, dmg_cats, uids)]
                df = pd.DataFrame(newvals, columns=newcols)
                ann_list.append(df)
            else:
                # Skip images with no annotations
                if os.path.exists(skipfile):
                    append_write = 'a'  # append if already exists
                else:
                    append_write = 'w'  # make a new file if not

                skipped = open(skipfile, append_write)
                skipped.write(os.path.basename(k) + '\n')
                skipped.close()
        anndf = pd.concat(ann_list, ignore_index=True)
        return anndf

    def generate_dmg_segmaps(self, split=1.0):
        """
        Generate segmentation maps for dataset.

        :return: None
        """
        assert split <= 1.0, "Invalid split"

        self.seg_dir = os.path.join(os.path.dirname(self.img_dir), 'cls_segmaps')
        # Create segementation directory
        if os.path.exists(self.seg_dir) is False:
            os.mkdir(self.seg_dir)
        else:
            shutil.rmtree(self.seg_dir)
            os.mkdir(self.seg_dir)

        # Create split directories
        tseg_dir = os.path.join(self.seg_dir, 'train')
        vseg_dir = os.path.join(self.seg_dir, 'val')

        if os.path.exists(tseg_dir) is False:
            os.mkdir(tseg_dir)
            os.mkdir(vseg_dir)
        else:
            shutil.rmtree(tseg_dir)
            shutil.rmtree(vseg_dir)
            os.mkdir(tseg_dir)
            os.mkdir(vseg_dir)

        # Get names i.e. palu-tsuname_000001
        _ids = set(['_'.join(os.path.basename(i).split('_')[0:2]) for i in self.jsons])
        t_ids = random.sample(_ids, int(split * len(_ids)))
        v_ids = list(set(_ids) - set(t_ids))

        train_jsons = []
        val_jsons = []

        pre_ext = "_pre_disaster.json"
        post_ext = "_post_disaster.json"
        json_dir = os.path.dirname(self.jsons[0])

        for tj in t_ids:
            pre_path = os.path.join(json_dir, tj + pre_ext)
            post_path = os.path.join(json_dir, tj + post_ext)
            train_jsons.append(pre_path)
            train_jsons.append(post_path)
        for vj in v_ids:
            pre_path = os.path.join(json_dir, vj + pre_ext)
            post_path = os.path.join(json_dir, vj + post_ext)
            val_jsons.append(pre_path)
            val_jsons.append(post_path)

        assert len(self.jsons) == len(train_jsons) + len(val_jsons)

        for j in tqdm(train_jsons):
            segmap = utils.generate_segmap(self.anndf, j)
            im = Image.fromarray(segmap)
            im.save(os.path.join(tseg_dir, os.path.basename(j)[:-5]) + ".png")

        for j in tqdm(val_jsons):
            segmap = utils.generate_segmap(self.anndf, j)
            im = Image.fromarray(segmap)
            im.save(os.path.join(vseg_dir, os.path.basename(j)[:-5]) + ".png")

    def generate_coco(self, split=1.0):
        """
        Convert annotations to MS-COCO format.

        :return: None
        """
        assert split <= 1.0, "Invalid split"

        # Create a coco directory
        coco_dir = "./coco"
        if os.path.exists(coco_dir):
            shutil.rmtree(coco_dir)
            os.mkdir(coco_dir)
        else:
            os.mkdir(coco_dir)

        if split == 1.0:
            # No split
            utils.generate_coco(self.predf, self.postdf, coco_dir)
        else:
            # Perform split -> train = split, val = (1 - split)
            utils.generate_coco_split(self.predf, self.postdf, split, coco_dir)

    def pre_post_split(self):
        """
        Generate pre-disaster and post-disaster dataframes from main dataframe.

        :return: pre disaster and post disaster dataframes
        """
        self.predf = self.anndf.loc[self.anndf["type"] == 'pre']
        self.postdf = self.anndf.loc[self.anndf["type"] == 'post']
        return self.predf, self.postdf

    def view_pre_post(self, disaster="guatemala-volcano", imid="00000000"):
        """
        Visualise the effect of a disaster via pre and post disaster images.

        :param disaster: Disaster name from the following list:
        ['guatemala-volcano', 'hurricane-florence', 'hurricane-harvey',
        'hurricane-matthew', 'hurricane-michael', 'mexico-earthquake',
        'midwest-flooding', 'palu-tsunami', 'santa-rosa-wildfire','socal-fire']

        :param imid: img id
        :return: None
        """
        assert disaster in self.anndf.disaster.unique()

        pre_df = self.predf[(self.predf['img_id'] == imid) &
                            (self.predf['disaster'] == disaster)]
        post_df = self.postdf[(self.postdf['img_id'] == imid) &
                              (self.postdf['disaster'] == disaster)]

        assert len(pre_df) + len(post_df) == len(self.anndf[(self.anndf['disaster'] == disaster) & (self.anndf['img_id'] == imid)])

        fig, axes = plt.subplots(1, 2, figsize=(15, 15), sharey=True)

        # Get pre and post disaster images
        pre_im = plt.imread(os.path.join(self.img_dir, pre_df.img_name.unique()[0]))
        post_im = plt.imread(os.path.join(self.img_dir, post_df.img_name.unique()[0]))
        axes[0].imshow(pre_im)
        axes[1].imshow(post_im)

        # Get pre-disaster building polygons
        for index, row in pre_df.iterrows():
            poly = shapely.wkt.loads(row['pixwkt'])
            dmg_stat = row['dmg_cat']
            axes[0].plot(*poly.exterior.xy, color='c')

        # Get post-disaster building polygons
        for index, row in post_df.iterrows():
            poly = shapely.wkt.loads(row['pixwkt'])
            dmg_stat = row['dmg_cat']
            axes[1].plot(*poly.exterior.xy, color=self.colordict[dmg_stat])

        axes[0].title.set_text('Pre Disaster')
        axes[0].axis('off')
        axes[1].title.set_text('Post Disaster')
        axes[1].axis('off')

        plt.suptitle(disaster + "_" + imid, fontsize=14, fontweight='bold')

        plt.show()

    def show_anns(self, ann):
        """
        Display the specified annotations.

        :param ann (str): location of annotation file to display
        :return: None
        """
        fname = os.path.basename(ann)[:-5]
        img_name = fname + ".png"
        img_path = os.path.join(self.img_dir, img_name)
        imgfile = plt.imread(img_path)

        # Get polygons
        plist, dmg = utils.get_object_polygons(self.anndf, ann, _type="poly")
        print("Number of objects =", len(plist))

        plt.figure(figsize=(15, 15))
        ax = plt.gca()
        polygons = []
        color = []

        # As long as plist is non-empty, add polys to axes
        if plist:
            for p, d in zip(plist, dmg):
                c = self.colordict[d]
                polygons.append(Polygon(p[0]))
                color.append(c)
                p = PatchCollection(polygons,
                                    facecolor=color,
                                    linewidths=0,
                                    alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons,
                                edgecolors=color,
                                facecolor='none',
                                linewidths=2)
            ax.add_collection(p)
        ax.axis('off')
        ax.set_title(os.path.basename(ann[:-5]))
        ax.imshow(imgfile)
        plt.show()


if __name__ == "__main__":
    data_dir = "./data"
    folder = "mini"
    img_dir = os.path.join(data_dir, folder, 'images')
    lbl_dir = os.path.join(data_dir, folder, 'labels')
    xview = XView2(img_dir, lbl_dir)
    #xview.show_anns('./data/mini/labels/palu-tsunami_00000001_post_disaster.json')
    #xview.view_pre_post('palu-tsunami', '00000001')
    #xview.generate_dmg_segmaps(0.8)
    xview.generate_coco(0.75)
