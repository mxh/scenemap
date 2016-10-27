# Scene Structure Inference through Scene Map Estimation
This repository contains the code used to train and test the network described
in the paper [*Scene Structure Inference through Scene Map Estimation*](http://geometry.cs.ucl.ac.uk/projects/2016/scenemap/paper_docs/scenemap.pdf).

*Note*: The current data available is the data used to generate Figure 8 in the
paper, i.e. the overly dense and cluttered scenes. This means the performance
you will get is close to what you see in Table 2, not the one in Table 1. The
data for Table 1 will be released soon.

### Introduction
Currently, we release the model files for the depth, semantics and scene map
estimation networks, as well as a set of data generated using the rendering
pipeline described in the paper. We will be releasing the rendering code in the
future as well.

### License
This software is released under the MIT License (refer to the LICENSE file for
details).

### Requirements: software
* [Torch](https://github.com/torch/torch7/)
* [CuDNN]

### Setup

1. Clone the repository
```Shell
git clone https://github.com/mxh/scenemap.git
```
2. Download the model and data files
```Shell
# The directory in which the repo has been cloned is $SM_ROOT
cd $SM_ROOT
./download_models.sh
./download_data.sh
```

### Running the network
Each network (depth, semantics and scenemap) can be run using their respective
shell scripts:
```Shell
# depth
./run_depth.sh
# semantics
./run_semantics.sh
# scene map
./run_map.sh
```
The output shows performance, and the output can be judged qualitatively in 
test\_\*.png.
