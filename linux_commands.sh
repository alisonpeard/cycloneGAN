# start micromamba in linux
./bin/micromamba shell init -s bash -p ~/micromamba
source ~/.bashrc

# Anaconda ImportError: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:micromamba/envs/tensorflow/lib