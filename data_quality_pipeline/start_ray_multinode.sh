#!/bin/bash

MASTER_PORT=8899
CONDA_NAME=fair

module load proxy/proxy_20
source ~/.bashrc >/dev/null 2>&1
conda activate $CONDA_NAME >/dev/null 2>&1

NODE_LIST=($(sort "$PBS_NODEFILE" | uniq))

MASTER_ADDR=$(getent hosts ${NODE_LIST[0]} | awk '{print $1}')

for i in "${!NODE_LIST[@]}"; do

    node=${NODE_LIST[$i]} 
    
    node_ip=$(getent hosts $node | awk '{print $1}')

    if ((i == 0)); then
        
        ssh $node "module load proxy/proxy_20; \
            nohup $HOME/miniconda3/envs/$CONDA_NAME/bin/ray start \
            --head --port=$MASTER_PORT --num-cpus=48 \
            --num-gpus=4 --node-ip-address=$node_ip --verbose --block" &
    
    else
        
        ssh $node "module load proxy/proxy_20; \
            nohup $HOME/miniconda3/envs/$CONDA_NAME/bin/ray start \
            --address='$MASTER_ADDR:$MASTER_PORT' --num-cpus=48 \
            --num-gpus=4 --node-ip-address=$node_ip --verbose --block" &

    sleep 10s
    fi
    
done
