# Range-Aware Attention Network for LiDAR-based 3D Object Detection with Auxiliary Density Level Estimation

<img src="https://github.com/anonymous0522/RAAN/blob/master/docs/motivation.PNG" width="100%" height="100%">

Distributed Train:
~~~
python -m torch.distributed.launch —nproc_per_node=NUM_OF_GPU tools/train.py PATH_TO_CONFIG —work_dir PATH_TO_WORK_DIR
~~~
Normal Train:
~~~
python  tools/train.py PATH_TO_CONFIG —work_dir PATH_TO_WORK_DIR
~~~
Load and fine tune:
~~~
python3 tools/train.py PATH_TO_CONFIG --work_dir PATH_TO_WORK_DIR --load_from PATH_TO_MODEL
~~~
Test with test set:
~~~
python tools/dist_test.py PATH_TO_CONFIG —work_dir TPATH_TO_WORK_DIR --checkpoint PATH_TO_MODEL --testset —speed_test
~~~
With validation set:
~~~
python tools/dist_test.py PATH_TO_CONFIG —work_dir TPATH_TO_WORK_DIR --checkpoint PATH_TO_MODEL —speed_test
~~~
With distributed val:
~~~
python -m torch.distributed.launch —nproc_per_node=NUM_OF_GPU tools/dist_test.py PATH_TO_CONFIG —work_dir TPATH_TO_WORK_DIR --checkpoint PATH_TO_MODEL --testset —speed_test
~~~

