## Codes for Conducting Experiments ##

conda activate DST_old
CUDA_VISIBLE_DEVICES=0 python main.py ../configs/KAGE_GPT2_FullTraining.jsonnet --mode train --experiment_name KAGE_DS_L4P4K2 --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part
CUDA_VISIBLE_DEVICES=0 python main.py ../configs/KAGE_GPT2_FullTraining.jsonnet --mode test --experiment_name KAGE_DS_L4P4K2 --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part --test_evaluation_name epoch6 --load_epoch 6
CUDA_VISIBLE_DEVICES=0 python main.py ../configs/KAGE_GPT2_FullTraining.jsonnet --mode test --experiment_name KAGE_DS_L4P4K2 --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part --test_evaluation_name epoch7 --load_epoch 7
CUDA_VISIBLE_DEVICES=0 python main.py ../configs/KAGE_GPT2_FullTraining.jsonnet --mode test --experiment_name KAGE_DS_L4P4K2 --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part --test_evaluation_name epoch8 --load_epoch 8

conda activate DST_old
CUDA_VISIBLE_DEVICES=2 python main.py ../configs/KAGE_GPT2_SparseSupervision.jsonnet --mode train --experiment_name KAGE_DS_L4P4K2_LastTurn --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part --only_last_turn
CUDA_VISIBLE_DEVICES=2 python main.py ../configs/KAGE_GPT2_SparseSupervision.jsonnet --mode test --experiment_name KAGE_DS_L4P4K2_LastTurn --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part --only_last_turn --test_evaluation_name epoch21 --load_epoch 21
CUDA_VISIBLE_DEVICES=2 python main.py ../configs/KAGE_GPT2_SparseSupervision.jsonnet --mode test --experiment_name KAGE_DS_L4P4K2_LastTurn --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part --only_last_turn --test_evaluation_name epoch24 --load_epoch 24
