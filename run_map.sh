#!/bin/bash
cd main
th main.lua -nGPU 1 -batch_size 32 -buffer_size 1000 -dataset mapfixed -net_type direct_l1 -grid_size 16 -test_only true -resume ../models/map -storage_dir ../data -postprocess non_max_suppression -test_image test_map.png
mv test_map.png ..
