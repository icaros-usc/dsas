import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#plt.rcParams["pdf.fonttype"] = 42
#plt.rcParams["ps.fonttype"] = 42

BC_type = "BC1_new"
data_folder = str("../experiments/"+BC_type)
data = pd.read_csv(data_folder + '/summary.csv')

y_label = "QD-Score"

plt.figure(figsize = (12,12))

sns.set(font_scale=4)
with sns.axes_style("white"):
	sns.set_style("white",{'font.family':'serif','font.serif':'Palatino'})
	# Plot the responses for different events and regions
	sns_plot = sns.lineplot(x="Evaluations", y=y_label,        hue="Algorithm", data=data)
	#sns.set_style()
	plt.xticks([0, 5000, 10000])
	#plt.yticks([0, 25], fontsize=20) #8Binary
	plt.yticks([0, 15000])  #KL
	#plt.yticks([0, 500, 1000], fontsize=20)  #MarioGAN
	plt.xlabel("Evaluations")
	plt.ylabel(y_label)

	if BC_type == "BC1_new":
 	  plt.title("BC1 & BC3", y = 1.08)
	elif BC_type == "BC2":
	  plt.title("BC1 & BC2, 2 goals", y = 1.08)
	elif BC_type == "BC3":		
	  plt.title("BC1 & BC2, 3 goals",y = 1.08)

	sns_plot.yaxis.set_label_coords(-0.05,0.5)
	sns_plot.xaxis.set_label_coords(0.5,-0.15)

	legend = plt.legend(loc='best',frameon=False, prop={'size': 30})
	frame = legend.get_frame()
	frame.set_facecolor('white')
	legend.remove()
	plt.tight_layout()
	plt.show()
	sns_plot.figure.savefig(BC_type+str(".pdf"))
