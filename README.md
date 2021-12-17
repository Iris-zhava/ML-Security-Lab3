# ML-Security-Lab3

## How to run
`python pruning.py`

## Load Data and Test Model
`python eval.py clean_data_filename poisoned_data_filename model_filename`

## Pruning Model
I prune the last pooling layer (indeed, the convolutional layer before the pooling layer), one channel at a time.
Channels are removed in decreasing order of activation values computed over the whole validation set of clean images.
In fact I also pruned them in increasing order as well, and I will give both results in the following.

### How to prune a single channel?
By setting the weight of the corresponding channel of the previous convolutional layer to zero. The bias parameters of corresponding channel should be set to zero as well.
In this way, no matter what the input feature map is, the output result will be zero after pruning, thus we create a "dead" channel to achieve the goal of pruning.

## Good Net
The goodnet is designed as follows: 
For each test input, run it through both unpruned model and pruned model. If the predicted results are the same, we output the result. Otherwise, two networks cannot
achieve agreements, then we consider the input to be a backdoored poisoned data, and we will output class N+1 to indicate the backdoor.

## Accuracy and Attack Success Rate against pruned channel numbers
If we prune the channels in decreasing order of activations:
![dec](https://github.com/Iris-zhava/ML-Security-Lab3/blob/main/pruning_descend.png)

If we prune the channels in increasing order of activations:
![dec](https://github.com/Iris-zhava/ML-Security-Lab3/blob/main/pruning_ascend.png)
