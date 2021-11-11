# Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking
This is the official repository of EMNLP 2021 paper "Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking".
The paper was presented at EMNLP 2021, which is now available at [https://aclanthology.org/2021.emnlp-main.620/](https://aclanthology.org/2021.emnlp-main.620/).

## install requirements
```bash
pip install -r Src/requirements.txt
```
## Prepare Data
### MultiWOZ data
Download the file and unzip to the project folder.

[Data(Google Drive)](https://drive.google.com/file/d/1utytDe3ojKPmDRBQgvm4gW_7Gn3AreBL/view?usp=sharing)

### Pre-trained models
Download and extract folders to `./Experiments`

[Pre-trained KAGE-GPT2 models(Google Drive)](https://drive.google.com/file/d/1ywtOIuQEA1W94mh21klJPM79QL6lv7BS/view?usp=sharing)

[Pre-trained DSTQA models(Google Drive)](https://drive.google.com/file/d/19Dbaiki_dHzUI-PTZBirnjaHInC4HXxB/view?usp=sharing)


## Guide
### Trainiing
#### KAGE Model (DSGraph + L4P4K2)
Training:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py ../configs/KAGE_GPT2_FullTraining.jsonnet --mode train --experiment_name KAGE_DS_L4P4K2 --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part
```
Testing:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py ../configs/KAGE_GPT2_FullTraining.jsonnet --mode test --experiment_name KAGE_DS_L4P4K2 --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part --test_evaluation_name epoch8 --load_epoch 8
```
#### KAGE Model (DSGraph + L4P4K2 + Sparse supervision)
Training:
```bash
CUDA_VISIBLE_DEVICES=2 python main.py ../configs/KAGE_GPT2_SparseSupervision.jsonnet --mode train --experiment_name KAGE_DS_L4P4K2_LastTurn --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part --only_last_turn
```
Testing:
```bash
CUDA_VISIBLE_DEVICES=2 python main.py ../configs/KAGE_GPT2_SparseSupervision.jsonnet --mode test --experiment_name KAGE_DS_L4P4K2_LastTurn --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part --only_last_turn --test_evaluation_name epoch24 --load_epoch 24
```
### Arguments
```
usage: main.py [-h] [--DATA_FOLDER DATA_FOLDER]
               [--EXPERIMENT_FOLDER EXPERIMENT_FOLDER] [--disable_cuda]
               [--device DEVICE] [--mode MODE] [--reset] [--only_last_turn]
               [--dummy_dataloader] [--experiment_name EXPERIMENT_NAME]
               [--fp16] [--load_best_model] [--load_epoch LOAD_EPOCH]
               [--load_model_path LOAD_MODEL_PATH]
               [--test_num_evaluation TEST_NUM_EVALUATION]
               [--test_batch_size TEST_BATCH_SIZE]
               [--test_num_processes TEST_NUM_PROCESSES]
               [--test_evaluation_name TEST_EVALUATION_NAME]
               [--test_disable_plot_img] [--test_output_attention]
               [--num_head NUM_HEAD] [--num_layer NUM_LAYER]
               [--num_hop NUM_HOP] [--graph_mode GRAPH_MODE]
               config_json_file

Knowledge-aware graph-enhanced GPT-2 for DST

positional arguments:
  config_json_file      The Configuration file in json/jsonnet format

optional arguments:
  -h, --help            show this help message and exit
  --DATA_FOLDER DATA_FOLDER
                        The path where the data is saved.
  --EXPERIMENT_FOLDER EXPERIMENT_FOLDER
                        The path where the experiments will be saved.
  --disable_cuda        Disable CUDA, run on CPU.
  --device DEVICE       Which CUDA device to use. Device ID.
  --mode MODE           train/test, see README.md for more details.
  --reset               This flag will try to delete already generated
                        experiment data.
  --only_last_turn      Switch to use sparse supervision, 14.3 percent of data
  --dummy_dataloader    Use only a small portion of data to run the program.
                        Useful for debugging.
  --experiment_name EXPERIMENT_NAME
                        The experiment name of the current run.
  --fp16                Not used.
  --load_best_model     Whether to load best model for testing/continue
                        training.
  --load_epoch LOAD_EPOCH
                        Specify which epoch of model to load from.
  --load_model_path LOAD_MODEL_PATH
                        Specify a path to a pre-trained model.
  --test_num_evaluation TEST_NUM_EVALUATION
                        How many data entries need to be tested.
  --test_batch_size TEST_BATCH_SIZE
                        Batch size of test.
  --test_num_processes TEST_NUM_PROCESSES
                        0 to disable multiprocessing testing; default is 4.
  --test_evaluation_name TEST_EVALUATION_NAME
                        Evaluation name which will be created at
                        /path/to/experiment/test/$test_evaluation_name$
  --test_disable_plot_img
                        Not used.
  --test_output_attention
                        For extracting attention scores. No effect for
                        reproduction.
  --num_head NUM_HEAD   Number of attention heads of GATs
  --num_layer NUM_LAYER
                        Number of GAT layers
  --num_hop NUM_HOP     Number of GAT hops.
  --graph_mode GRAPH_MODE
                        part: DSGraph; full: DSVGraph
```

## (Baseline) DSTQA
This repository also contains code for reproducing DSTQA model

DSTQA was proposed in the following paper:

Li Zhou, Kevin Small. Multi-domain Dialogue State Tracking as Dynamic Knowledge Graph Enhanced Question Answering. In NeurIPS 2019 Workshop on Conversational AI ([PDF](https://arxiv.org/pdf/1911.06192.pdf))

### Prepare Data
Step 1 - Install Requirements
```
pip install -r Src/requirements_DSTQA.txt
```
Step 2 - Download Dataset
```
cd Materials/DSTQA/
wget https://raw.githubusercontent.com/jasonwu0731/trade-dst/master/create_data.py
wget https://raw.githubusercontent.com/jasonwu0731/trade-dst/master/utils/mapping.pair
sed -i 's/utils\/mapping.pair/mapping.pair/g' create_data.py
python create_data.py 
```

Step 3 - Preprocess Dataset
```
python multiwoz_format.py all ./data ./data
```

Step 4 - Pre-calculate ELMO Embeddings
```
mkdir ./data/elmo_embeddings
bash calc_elmo.sh ./data ./data/elmo_embeddings
```

### Full Supervision
```bash
python main.py ../configs/DSTQA.jsonnet --mode train --experiment_name DSTQA_Baseline_new
python main.py ../configs/DSTQA.jsonnet --mode test --experiment_name DSTQA_Baseline_new --test_evaluation_name epoch109 --load_epoch 109
```

### Sparse Supervision
First go to `Src/models/dstqa/dstqa.py Line 230-234`:
```
## Switch to sparse supervision
# (1) Full Training
# sampled_turn = random.choice(list(range(max_turn_count)))
# (2) Sparse Supervision
sampled_turn = max_turn_count-1  # Use last turn only
```
Then run training and testing:
```bash
python main.py ../configs/DSTQA.jsonnet --mode train --experiment_name DSTQA_LastTurn
python main.py ../configs/DSTQA.jsonnet --mode test --experiment_name DSTQA_LastTurn --test_evaluation_name epoch114 --load_epoch 114
```

## Results
Below attached the results you expect to get from the pre-trained models:
| Model                                      | Joint Accuracy | Slot Accuracy | Epoch |
|--------------------------------------------|----------------|---------------|-------|
| DSTQA                                      | 0.5224         | 0.9728        | 174   |
| DSTQA-SparseSupervision                    | 0.2288         | 0.9353        | 109   |
| KAGE-GPT2-DSGraph-L4P4K2                   | 0.5515         | 0.9746        | 7     |
| KAGE-GPT2-DSGraph-L4P4K2-SparseSupervision | 0.5023         | 0.9707        | 24    |

Please note that the models released here are reproductions after the publication of our paper.

The numbers reported in the paper are the average of 3 runs, which might differ slightly from what were reported here.

The reproduced models were trained and tested on NVIDIA V100 Clusters, not on the original NVIDIA GTX 3090.

## Citation
If our research helps you, please kindly cite our paper:
```
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP).
```
```
@inproceedings{lin-etal-2021-knowledge,
    title = "Knowledge-Aware Graph-Enhanced {GPT}-2 for Dialogue State Tracking",
    author = "Lin, Weizhe  and
      Tseng, Bo-Hsiang  and
      Byrne, Bill",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.620",
    pages = "7871--7881",
    abstract = "Dialogue State Tracking is central to multi-domain task-oriented dialogue systems, responsible for extracting information from user utterances. We present a novel hybrid architecture that augments GPT-2 with representations derived from Graph Attention Networks in such a way to allow causal, sequential prediction of slot values. The model architecture captures inter-slot relationships and dependencies across domains that otherwise can be lost in sequential prediction. We report improvements in state tracking performance in MultiWOZ 2.0 against a strong GPT-2 baseline and investigate a simplified sparse training scenario in which DST models are trained only on session-level annotations but evaluated at the turn level. We further report detailed analyses to demonstrate the effectiveness of graph models in DST by showing that the proposed graph modules capture inter-slot dependencies and improve the predictions of values that are common to multiple domains.",
}
```


