#!/bin/bash
cd main
th main.lua -nGPU 1 -batch_size 32 -buffer_size 1000 -dataset depthfixed -net_type depth -test_only true -resume ../models/depth -storage_dir ../data -test_image test_dep.png
