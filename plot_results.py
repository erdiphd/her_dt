import re
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
print(sns.__version__)
legend_list = []


def plot_graphs(file_list, color='r'):
    success_result_array = []
    global legend_list
    for file_number in range(len(file_list)):
        logs = open(file_list[file_number])
        logs.seek(0)
        log_file = logs.read()
        if file_list[0] == "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HGG/ddpg-FetchPush-v1-hgg-(2023-06-14-11:38:56).log" or file_list[0] == "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HER/ddpg-FetchPush-v1-normal-(2023-06-14-16 41 19).log":
            success_rates = [(m.start(0), m.end(0)) for m in re.finditer('Success/obstacle', log_file)]
        else:
            success_rates = [(m.start(0), m.end(0)) for m in re.finditer('Success/interval', log_file)]

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
file_list = [ "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPickAndPlace/ddpg-FetchPickAndPlace-v1-hgg_dt-(2023-06-15-13:21:10).log",

"/home/vlaffo/Desktop/thesis results/obstacle test/FetchPickAndPlace/ddpg-FetchPickAndPlace-v1-hgg_dt-(2023-06-17-15:46:04).log",

 "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPickAndPlace/ddpg-FetchPickAndPlace-v1-hgg_dt-(2023-06-18-20:12:05).log",

 # "/home/vlaffo/Desktop/thesis results/obstacle test/FetchSlide/ddpg-FetchSlide-v1-hgg_dt-(2023-06-19-18:35:31).log",

 # "/home/vlaffo/Desktop/thesis results/obstacle test/FetchSlide/HGG-DT/ddpg-FetchPush-v1-hgg_dt-obstacle.log",
              ]
plot_graphs(file_list, 'red')

# HGG
# file_list = [ "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HGG/ddpg-FetchPush-v1-hgg-(2023-06-14-11:38:56).log",
#  "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HGG/ddpg-FetchPush-v1-hgg-(2023-06-14-11:39:14).log",
#
#  "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HGG/ddpg-FetchPush-v1-hgg-(2023-06-14-11:39:34).log",
#
# "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HGG/ddpg-FetchPush-v1-hgg-(2023-06-14-11:39:46).log",
#
#  "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HGG/ddpg-FetchPush-v1-hgg-obstacle.log",
#             ]
# plot_graphs(file_list, 'blue')
#
# # HER
# file_list = [ "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HER/ddpg-FetchPush-v1-normal-(2023-06-14-16 41 19).log",
#  "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HER/ddpg-FetchPush-v1-normal-(2023-06-14-16 52 43).log",
#
#  "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HER/ddpg-FetchPush-v1-normal-(2023-06-14-16 52 45).log",
#
# "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HER/ddpg-FetchPush-v1-normal-(2023-06-14-16 52 49).log",
#
#  "/home/vlaffo/Desktop/thesis results/obstacle test/FetchPush/HER/ddpg-FetchPush-v1-normal-(2023-06-14-16 52 56).log",
#             ]
# plot_graphs(file_list, 'green')


sns.tsplot(time=xdata, data=np.ones(len(xdata)), color="b", linestyle="-")

plt.ylabel("Success Rate", fontsize=15)
plt.xlabel("Episode ", fontsize=15, labelpad=4)
plt.title('FetchPickAndPlace + obstacle', fontsize=16)

plt.legend(labels=["DT-HER"],
           loc='lower left', )
# plt.legend(labels=legend_list)
plt.show()

# logs.close()
