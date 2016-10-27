#!/bin/bash
cd main
th main.lua -nGPU 1 -batch_size 32 -buffer_size 1000 -dataset semfixed -net_type semantics -test_only true -resume ../models/semantics -storage_dir ../data -test_image test_sem.png
