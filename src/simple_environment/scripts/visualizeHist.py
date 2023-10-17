#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata
import seaborn as sns
import toml
import argparse
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm


from simple_environment.util.SearchHelper import *
from simple_environment.util.bc_calculate import *

#file_name = "RANDOM_BC_sim4all_simulations.csv"

#file_name = "MAPELITES_BC_sim0all_simulations.csv"
#file_name = "RANDOM_BC_sim4all_simulations.csv"

file_name = "CMAME_BC_sim13all_simulations.csv"

#file_name = "CMAME_BC_sim12all_simulations.csv"


BC_TYPE = "BC2"
ALG_TYPE = "CMAME"
#ALG_TYPE = "CMAME"
folder_name = str(BC_TYPE + "/" + ALG_TYPE + "/")
MAP_SIZE_1 = 25
MAP_SIZE_2 = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-c','--config', help='path of BC config file',required=True)
    opt = parser.parse_args()

    parser = argparse.ArgumentParser()
    elite_map_config=toml.load(opt.config)
    feature_ranges = []
    column_names = []
    bc_names = []
    for bc in elite_map_config["Map"]["Features"]:
      feature_ranges.append((bc["low"],bc["high"]))
      column_names.append(bc["name"])
      bc_names.append(bc["name"])

    feature1Range = feature_ranges[0]
    feature2Range = feature_ranges[1]

    feature_map = FeatureMap(-1, feature_ranges, resolutions = (MAP_SIZE_1, MAP_SIZE_2))

    with open("../output/"+folder_name + file_name) as csvfile:
        all_records = csv.reader(csvfile, delimiter=',')
        hist = np.zeros((MAP_SIZE_1,MAP_SIZE_2))
        #from IPython import embed
        #embed()
        all_records.next()

        for data_point in enumerate(all_records):
                if data_point!='':
                    #i=i+1
                    #data_point_info=data_point[1:-1]
                    #data_point_info=data_point.split(', ')
                    data_point_info = data_point[1]
                    #data_point=data_point[1:-1]
                    
                    #data_point_info=data_point.split(', ')
                    #from IPython import embed
                    #embed()
                    feature1Range = feature_ranges[0]
                    feature2Range = feature_ranges[1]
                    #data_point_info=data_point.split(', ')

                    feature1 = float(data_point_info[12])
                    feature2 = float(data_point_info[13])

                    cell_x = feature_map.get_feature_index(0, feature1)
                    cell_y = feature_map.get_feature_index(1, feature2)
                    #cell_x = gridsize-1 - int(cell_x*gridsize)
                    hist[cell_x, cell_y] = hist[cell_x, cell_y] + 1


    plt.figure()
    print(hist)

    histMirr = np.zeros([MAP_SIZE_1,MAP_SIZE_2])
    for i in range(0,MAP_SIZE_1):
        for j in range(0,MAP_SIZE_2):
            histMirr[MAP_SIZE_1 -1-i,j] = hist[i,j]

    hist = histMirr

    cmap = ListedColormap(sns.light_palette("navy",100))

    with sns.axes_style("white"):
        #numTicks = 10#11
        plt.figure(figsize=(12,12))
        sns.set(font_scale=4)
        sns.set_style({'font.family':'serif', 'font.serif':'Palatino'})
        #map = np.flip(map,0)
        mask = np.zeros_like(hist)
        zero_indices = np.where(hist == 0)
        hist[zero_indices] = np.nan
        cmap.set_bad("white") 

        #showing elite   
        maxval = 30

   

        g = sns.heatmap(hist, annot=False, fmt=".0f",
                   #yticklabels=[],
                vmin=1.0,
                vmax=maxval,
                mask = mask,
                cmap = cmap,
                rasterized=True,
                #square = True,
                cbar = False)
                #vmin=np.nanmin(fitnessMap),
                #vmax=np.nanmax(fitnessMap))
        fig = g.get_figure()
        g.set(xticks = [0,MAP_SIZE_2-1])
        g.set(xticklabels = ['0','0.11'])
        g.set(yticks = [0,MAP_SIZE_1])
        g.set(yticklabels = ['0.32','0'])
        for item in g.get_xticklabels():
            item.set_rotation(0)
        for item in g.get_yticklabels():
            item.set_rotation(0)
        if BC_TYPE == "BC2" or BC_TYPE == "BC3":
            g.set(xlabel = "Variation")
            g.set(xticklabels=['0','0.11'])
        else:
            g.set(xlabel = r"$\beta$")
            g.set(xticklabels=['0','1000'])
        if ALG_TYPE == "ME":
            plt.title('MAP-Elites')
        elif ALG_TYPE == "CMAES":
            plt.title('CMA-ES')
        elif ALG_TYPE == "RANDOM":
            plt.title("RANDOM")
        elif ALG_TYPE == "CMAME":
            plt.title("CMA-ME")
        g.set(ylabel = "Distance")
        g.yaxis.set_label_coords(-0.05,0.5)
        g.xaxis.set_label_coords(0.5,-0.05)
        matplotlib.rcParams.update({'font.size': 20})
        plt.show()
        plt.tight_layout()
        g.figure.savefig(str("heat_" + BC_TYPE+"_" + ALG_TYPE))

    #from IPython import embed
