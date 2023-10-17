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
#file_name="MAPELITES_BC_sim122_elites_freq1.csv"
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

from simple_environment.util.SearchHelper import *
from simple_environment.util.bc_calculate import *

#BC2-CMAES-file_name = "CMAES_BC_sim2_elites_freq20.csv"
#BC2-ME-file_name = "MAPELITES_BC_sim0_elites_freq20.csv"
#BC2-RANDOM-file_name = "RANDOM_BC_sim4_elites_freq20.csv"

#5 83

#file_name = "MAPELITES_BC_sim0_elites_freq20.csv"
#file_name = "MAPELITES_BC_sim50001_elites_freq20.csv"
#file_name = "RANDOM_BC_sim3_elites_freq20.csv"
#file_name = "MAPELITES_BC_sim-6_elites_freq20.csv"
#file_name = "CMAES_BC_sim0_elites_freq20.csv"
#BC_TYPE = "BC3"
#file_name = "MAPELITES_BC_sim0_elites_freq20.csv"
#ALG_TYPE = "ME"


BC_TYPE = "BC2"

#file_name = "CMAME_BC_sim13_elites_freq20.csv"
#ALG_TYPE = "CMAME"
file_name = "RANDOM_BC_sim4_elites_freq20.csv"

#file_name = "MAPELITES_BC_sim0_elites_freq20.csv"
ALG_TYPE = "RANDOM"

folder_name = str(BC_TYPE + "/" + ALG_TYPE + "/")
#folder_name = "logs/"


#feature1Range = (0, 0.316)
#feature2Range = (0.212, 0.46)
MAP_SIZE_1 = 25
MAP_SIZE_2 = 100

def import_data_file(file_name,feature_ranges, map_threshold):
    QD_scores = [] 
    coverages = [] 
    feature_map = FeatureMap(-1, feature_ranges, resolutions = (MAP_SIZE_1, MAP_SIZE_2))
    with open("../output/"+folder_name + file_name) as csvfile:
    #with open("src/simple_environment/"+folder_name + file_name) as csvfile:
            all_records = csv.reader(csvfile, delimiter=',')
            print "importing data from file: " + str(file_name)
            for i,one_map in enumerate(all_records):
                #print(i)
                #if i > 300: 
                #  break
                fitnesses = []
                num_cells = 0
                map = np.zeros((MAP_SIZE_1, MAP_SIZE_2))
                if  i == map_threshold:
                    for data_point in one_map: 
                        if data_point!='':
                            data_point=data_point[1:-1]
                            #from IPython import embed
                            #embed()
                            feature1Range = feature_ranges[0]
                            feature2Range = feature_ranges[1]
                            data_point_info=data_point.split(', ')

                            #cell_x =(float(data_point_info[12]) - feature1Range[0])/(feature1Range[1]-feature1Range[0])
                            #cell_y = (float(data_point_info[13]) - feature2Range[0])/(feature2Range[1]-feature2Range[0])
                            #from IPython import embed 
                            #embed()
                            #print(data_point_info[12])
                            feature1 = float(data_point_info[12])
                            feature2 = float(data_point_info[13])
                            cell_x = feature_map.get_feature_index(0, feature1)
                            cell_y = feature_map.get_feature_index(1, feature2)
                            fitness = float(data_point_info[2])
                            #cell_x = gridsize-1 - int(cell_x*gridsize)
                            # if cell_x == 25-1-16 and cell_y ==85: 
                            #     #from IPython import embed
                            #     embed()
                            #  continue
                            map[cell_x, cell_y] = float(data_point_info[2])
                            cell_indx = int(data_point_info[0])
                            fitnesses.append(fitness)
                            num_cells = num_cells + 1

 
                if i == map_threshold:
                    #from IPython import embed
                    #embed()
                    print "i: " + str(i)
                    print "QD score: " + str(sum(fitnesses))
                    QD_scores.append(sum(fitnesses))
                    coverages.append(float(num_cells)/ (MAP_SIZE_1 * MAP_SIZE_2))
                    print "coverage: " + str(float(num_cells)/ (MAP_SIZE_1  * MAP_SIZE_2))
                    return map, QD_scores, coverages




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
    
    for iteration_indx in range(0,500):
	    
	    map, QD_scores_me, coverages_me = import_data_file(file_name,feature_ranges, iteration_indx)               

	    mapMirr = np.zeros([MAP_SIZE_1,MAP_SIZE_2])
	    for i in range(0,MAP_SIZE_1):
	        for j in range(0,MAP_SIZE_2):
	            mapMirr[MAP_SIZE_1 -1-i,j] = map[i,j]

	    map = mapMirr

	    cmap = ListedColormap(sns.color_palette("coolwarm", 6).as_hex())
	    with sns.axes_style("white"):
	        #numTicks = 10#11
	        plt.figure(figsize=(12,12))
	        sns.set(font_scale=4)
	        sns.set_style({'font.family':'serif', 'font.serif':'Palatino'})
	        #map = np.flip(map,0)
	        mask = np.zeros_like(map)
	        zero_indices = np.where(map == 0)
	        mask[zero_indices] = np.nan
	        cmap.set_bad("white") 

	        #showing elite   
	        maxval = 10
	  
	        # #showing elite   
	       # maxval = 14
	        #indx1 = 21
	        #indx2 = 52

	 #       map[MAP_SIZE_1-1-indx1][indx2] = maxval

	        # maxval = 14
	        # indx1 = 9
	        # indx2 = 83

	        # map[MAP_SIZE_1-1-indx1][indx2] = maxval
	        # feature1 = float(indx1)/(MAP_SIZE_1-1) * feature_ranges[0][1]
	        # feature2 = float(indx2)/(MAP_SIZE_2-1) * feature_ranges[1][1]
	        # print("feature1 is: " + str(feature1))
	        # print("feature2 is: " + str(feature2))     
	        
	   

	        g = sns.heatmap(map, annot=False, fmt=".0f",
	                   #yticklabels=[],
	                vmin=2.0,
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
	        g.set(yticks = [0,MAP_SIZE_1])
	        g.set(yticklabels = ['0.32','0'])
	        for item in g.get_xticklabels():
	            item.set_rotation(0)
	        for item in g.get_yticklabels():
	            item.set_rotation(0)
	        if BC_TYPE == "BC2" or BC_TYPE == "BC3":
	            g.set(xlabel = "Variation")
	            g.set(xticklabels = ['0','0.11'])
	        else:
	            g.set(xlabel = r"$\beta$")
	            g.set(xticklabels = ['0','1000'])
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
	        #plt.show()
	        plt.tight_layout()
	        filename_template = "../images-random/" +  str(BC_TYPE+"_" + ALG_TYPE+"_" + '{:05d}.png')
	        filename = filename_template.format(iteration_indx)
	        g.figure.savefig(filename)
