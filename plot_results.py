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
        success_rates = [(m.start(0), m.end(0)) for m in re.finditer('Success/interval', log_file)]
        # print(success_rates)

        success_result_array_tmp = []
        for start_pos, end_pose in success_rates:
            # print(float(log_file[end_pose+1: end_pose+7]))
            success_result_array_tmp.append(float(log_file[end_pose + 3: end_pose + 7]))

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


robotic_task = 'FetchPush'
xdata = 200 * np.arange(100)

neural_network_option = 'compare_nn'

file_list = [ "/home/erdi/Desktop/ddpg-FetchPush-v1-hgg_dt-3.log",
             "/home/erdi/Desktop/ddpg-FetchPush-v1-hgg_dt_4.log",
              ]
plot_graphs(file_list, 'red')


file_list = [ 
             "/home/erdi/Desktop/ddpg-FetchPush-v1-hgg_dt_5.log"
              ]
plot_graphs(file_list, 'blue')



sns.tsplot(time=xdata, data=np.ones(len(xdata)), color="b", linestyle="-")

plt.ylabel("Success Rate", fontsize=15)
plt.xlabel("Episode ", fontsize=15, labelpad=4)

plt.legend(labels=["HGG-DT","HGG"],
           loc='lower left', )
# plt.legend(labels=legend_list)
plt.show()

# logs.close()
