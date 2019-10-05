import os

import json
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import shapely.wkt
import time

import utils


class XView2():
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

        self.colordict = {'no-damage': 'w',
                          'minor-damage': 'darseagreen',
                          'major-damage': 'orange',
                          'destroyed': 'red',
                          'un-classified': 'b'}

    def generate_dataframe(self):
        """
        Generate main annotation dataframe.

        :return: anndf (pandas DataFrame)
        """
        ann_list = []
        for fname, ann in self.anndict.items():
            # Get features
            feature_type = []
            uids = []
            pixwkts = []
            geowkts = []
            dmg_cats = []
            imids = []

            for i in ann['features']['xy']:
                feature_type.append(i['properties']['feature_type'])
                uids.append(i['properties']['uid'])
                pixwkts.append(i['wkt'])
                if 'subtype' in list(i['properties'].keys()):
                    dmg_cats.append(i['properties']['subtype'])
                else:
                    dmg_cats.append("none")
                imids.append(fname.split('_')[1])

            for i in ann['features']['lng_lat']:
                geowkts.append(i['wkt'])

            # Get Metadata
            cols = list(ann['metadata'].keys())
            vals = list(ann['metadata'].values())

            newcols = ['obj_type', 'img_id', 'pixwkt', 'geowkt', 'dmg_cat', 'uid'] + cols
            newvals = [[f, _id, pw, gw, dmg, u] + vals for f, _id, pw, gw, dmg, u in zip(feature_type, imids, pixwkts, geowkts, dmg_cats, uids)]
            df = pd.DataFrame(newvals, columns=newcols)
            ann_list.append(df)
        anndf = pd.concat(ann_list, ignore_index=True)
        return anndf

    def get_object_polygons(self, ann, wkt_type="pixel", _type="wkt"):
        """
        Return object polygons & damage status.

        :param ann (str): location of annotation file to parse
        :param wkt_type (str): wkt version to be returned (pixel or geo)
        :param _type (str): format of polygons to be returned (wkt or list of lists)
        :return: None
        """
        fname = os.path.basename(ann)[:-5]
        img_name = fname + ".png"
        ann = self.anndf.loc[self.anndf["img_name"] == img_name]
        if wkt_type == "pixel":
            wktlist = ann['pixwkt'].tolist()
        elif wkt_type == "geo":
            wktlist = ann['geowkt'].tolist()
        dmg = ann['dmg_cat'].tolist()
        polylist = utils.wkt2list(wktlist)
        if _type == "wkt":
            return wktlist, dmg
        elif _type == "poly":
            return polylist, dmg

    def pre_post_split(self):
        """
        Generate pre-disaster and post-disaster dataframes from main dataframe.

        :return: pre disaster and post disaster dataframes
        """
        self.pre_dist_df = self.anndf.loc[self.anndf["dmg_cat"] == 'none']
        self.post_dist_df = self.anndf.loc[self.anndf["dmg_cat"] != 'none']
        return self.pre_dist_df, self.post_dist_df

    def view_pre_post(self, disaster="guatemala-volcano", imid="00000000"):
        """
        Visualise the effect of a disaster via pre and post disaster images.

        :param disaster (str): Disaster name from the following list:
        ['guatemala-volcano', 'hurricane-florence', 'hurricane-harvey',
        'hurricane-matthew', 'hurricane-michael', 'mexico-earthquake',
        'midwest-flooding', 'palu-tsunami', 'santa-rosa-wildfire','socal-fire']

        :param imid (str): img id
        :return: None
        """
        assert disaster in self.anndf.disaster.unique()

        pre_df = self.pre_dist_df[(self.pre_dist_df['img_id'] == imid) &
                                  (self.pre_dist_df['disaster'] == disaster)]
        post_df = self.post_dist_df[(self.post_dist_df['img_id'] == imid) &
                                    (self.post_dist_df['disaster'] == disaster)]

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
        plist, dmg = self.get_object_polygons(ann, _type="poly")
        print("Number of objects =", len(plist))

        plt.figure(figsize=(15, 15))
        ax = plt.gca()
        polygons = []
        color = []
        if len(plist) != 0:
            # Pre_disaster Images
            if dmg is None:
                for p in plist:
                    c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
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

            # Post_disaster Images
            else:
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
    folder = "train"
    img_dir = os.path.join(data_dir, folder, 'images')
    lbl_dir = os.path.join(data_dir, folder, 'labels')
    xview = XView2(img_dir, lbl_dir)
    xview.show_anns('./data/train/labels/palu-tsunami_00000002_post_disaster.json')
    xview.view_pre_post('palu-tsunami', '00000002')
