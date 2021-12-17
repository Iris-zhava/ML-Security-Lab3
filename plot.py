import matplotlib.pyplot as plt

blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
purple = '#9467bd'

def log2arr(filepath):
    f = open(filepath, 'r')
    str_arr = f.readlines()
    arr = []
    for line in str_arr:
        arr.append(float(line.strip()))
    return arr

cl_acc_log = "cl_acc_log.txt"
ps_acc_log = "ps_acc_log.txt"

cl_acc_arr = log2arr(cl_acc_log)
ps_acc_arr = log2arr(ps_acc_log)

channel_pruned = range(60)

fig = plt.figure(figsize=(10, 5.5))
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Rate',fontdict={'size':20})
ax1.set_xlabel('Channels Pruned', fontdict={'size':20})
ax1.plot(channel_pruned, cl_acc_arr, color=blue, label='Clean Classification Accuracy')
ax1.plot(channel_pruned, ps_acc_arr, color=red, label='Backdoor Attack Success')


fig.legend(fontsize=14)
plt.setp(ax1.get_xticklabels(), fontsize=14)
plt.setp(ax1.get_yticklabels(), fontsize=14)

imgname = 'pruning_descend.png'
plt.savefig(imgname, dpi=600)



cl_acc_log = "cl_acc_log_ascending.txt"
ps_acc_log = "ps_acc_log_ascending.txt"

cl_acc_arr = log2arr(cl_acc_log)
ps_acc_arr = log2arr(ps_acc_log)

channel_pruned = range(60)

fig = plt.figure(figsize=(10, 5.5))
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Rate',fontdict={'size':20})
ax1.set_xlabel('Channels Pruned', fontdict={'size':20})
ax1.plot(channel_pruned, cl_acc_arr, color=blue, label='Clean Classification Accuracy')
ax1.plot(channel_pruned, ps_acc_arr, color=red, label='Backdoor Attack Success')


fig.legend(fontsize=14)
plt.setp(ax1.get_xticklabels(), fontsize=14)
plt.setp(ax1.get_yticklabels(), fontsize=14)

imgname = 'pruning_ascend.png'
plt.savefig(imgname, dpi=600)