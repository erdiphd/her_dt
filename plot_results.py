import re
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from os import walk
print(sns.__version__)
legend_list = []


def plot_graphs(file_list, color='r'):
    success_result_array = []
    global legend_list
    for file_number in range(len(file_list)):
        logs = open(file_list[file_number])
        logs.seek(0)
        log_file = logs.read()
        # if file_list[0] == "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HGG-DT/her_result1-ddpg-FetchPush-v1-dt-her-(2023-06-27-16_29_25).log" or file_list[0] == "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HGG/her_result1-ddpg-FetchPush-v1-hgg-(2023-06-27-09_23_52).log" or file_list[0] == "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HER/her_result1-ddpg-FetchPush-v1-normal-(2023-06-27-09_23_29).log":
        success_rates = [(m.start(0), m.end(0)) for m in re.finditer('Success/obstacle', log_file)]
        # else:
        # success_rates = [(m.start(0), m.end(0)) for m in re.finditer('Success/interval', log_file)]

        # print(success_rates)

        success_result_array_tmp = []
        for start_pos, end_pose in success_rates:
            # print(float(log_file[end_pose+1: end_pose+7]))
            success_result_array_tmp.append(float(log_file[end_pose + 2: end_pose + 7]))

        success_result_array.append(success_result_array_tmp)

    xdata = 50 * np.arange(len(success_result_array_tmp))
    # xdata = xdata /50 / 20

    sns.tsplot(
        time=xdata,
        data=success_result_array,
        color=color,
        linestyle="-"
    )

    learn_text_location = [(m.start(0), m.end(0)) for m in re.finditer('learn', log_file)]
    buffer_type_text_location = [(m.start(0), m.end(0)) for m in re.finditer('buffer_type', log_file)]
    reward_compute_type_location = [(m.start(0), m.end(0)) for m in re.finditer('reward_compute_type', log_file)]
    label_text = log_file[learn_text_location[0][1] + 2:learn_text_location[0][1] + 5] + " " + log_file[
                                                                                               buffer_type_text_location[
                                                                                                   0][1] + 2:
                                                                                               buffer_type_text_location[
                                                                                                   0][1] + 7]

    legend_list.append(label_text)


robotic_task = 'FetchSlide'
xdata = 200 * np.arange(100)

neural_network_option = 'compare_nn'

# HGG_DT
file_list = []
path = "/home/vlaffo/Desktop/thesis results/" + "obstacle test/" + robotic_task +"/HGG-DT/2 cell centered after change/"

for (dirpath, dirnames, filenames) in walk(path):
    file_list.append(path + filenames[0])
    file_list.append(path + filenames[1])
    file_list.append(path + filenames[2])
    file_list.append(path + filenames[3])
    file_list.append(path + filenames[4])

plot_graphs(file_list, 'red')

# HGG
file_list = []
path = "/home/vlaffo/Desktop/thesis results/" + "obstacle test/" + robotic_task +"/HGG/2 cell centered/"

for (dirpath, dirnames, filenames) in walk(path):
    file_list.append(path + filenames[0])
    file_list.append(path + filenames[1])
    file_list.append(path + filenames[2])
    file_list.append(path + filenames[3])
    file_list.append(path + filenames[4])
plot_graphs(file_list, 'blue')

# HER
file_list = []
path = "/home/vlaffo/Desktop/thesis results/" + "obstacle test/" + robotic_task +"/HER/2 cell centered/"

for (dirpath, dirnames, filenames) in walk(path):
    file_list.append(path + filenames[0])
    file_list.append(path + filenames[1])
    file_list.append(path + filenames[2])
    file_list.append(path + filenames[3])
    file_list.append(path + filenames[4])

plot_graphs(file_list, 'green')


sns.tsplot(time=xdata, data=np.ones(len(xdata)), color="b", linestyle="-")

plt.ylabel("Success Rate", fontsize=15)
plt.xlabel("Episode ", fontsize=15, labelpad=4)
plt.title(robotic_task + " + obstacle", fontsize=16)

plt.legend(labels=["DT-HER", "HGG+EBP", "HER"],
           loc='lower left', )
# plt.legend(labels=legend_list)
plt.show()
# logs.close()
