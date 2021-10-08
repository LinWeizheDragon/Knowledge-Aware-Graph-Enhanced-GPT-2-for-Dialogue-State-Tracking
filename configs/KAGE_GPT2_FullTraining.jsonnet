local train_data_path = "../Data/MultiWOZ/dst/train_v2.0.json";
local validation_data_path = "../Data/MultiWOZ/dst/dev_v2.0.json";
local test_data_path = "../Data/MultiWOZ/dst/test_v2.0.json";
local elmo_embedding_path = "../Materials/DSTQA/data/elmo_embeddings/all/";
local dummy_train_data_path = "../Materials/DSTQA/data/dummy/train.json";
local dummy_validation_data_path = "../Materials/DSTQA/data/dummy/dev.json";
local dummy_elmo_embedding_path = "../Materials/DSTQA/data/elmo_embeddings/dummy/";
local domain_slot_list_path = "../Materials/DSTQA/ontology/domain_slot_list_nosp.txt";
local preprocessed_data_path = "../Data/preprocessed_2.0_GPT2/";
local ontology_path = "../Data/MultiWOZ/dst/ontology_v2.0.json";
local base_model = "gpt2";
local train_batch_size = 2;
local valid_batch_size = 2;
local valid_gen_batch_size = 1;
local test_batch_size = 1;
local save_interval = 5000;
local train_epochs = 8;
local lr = 6.25e-5;
local graph_lr = 8e-5;
local adam_epsilon = 1e-12;
local gradient_accumulation_steps = 15;
local gradient_clipping = 1000.0;
local warmup_steps = 0;
{
  "DATA_FOLDER": "",
  "EXPERIMENT_FOLDER": "",
  "TENSORBOARD_FOLDER": "",
  "platform_type": "pytorch",
  "model": "KAGE",
  "model_type": "KAGE",
  "use_graph": 1,
  'freeze_transformer': 0,
  "ignore_pretrained_weights": ["graph"],
  "experiment_name": "default_test",
  "model_config": {
    "base_model": base_model,
    "aug_method": "concat",
    "graph_mode": "part",
    "residual": 0,
    "cls_loss": 0,
    "connect_type": "ds_value_only",
    "graph_model": {
        "model_type": "GAT",
        "num_layer": 4,
        "num_head": 4,
        "feature_size": 768,
        "num_hop": 2,
        "dropout": 0.2,
    },
  },
  "data_loader": {
    "type": "data_loader_kage",
    "dummy_dataloader": 0,
    "additional": {
        "train_data_path": train_data_path,
        "validation_data_path": validation_data_path,
        "test_data_path": test_data_path,
        "ontology_path": ontology_path,
        "elmo_embedding_path": elmo_embedding_path,
        "domain_slot_list_path": domain_slot_list_path,
        "dummy_train_data_path": dummy_train_data_path,
        "dummy_validation_data_path": dummy_validation_data_path,
        "dummy_elmo_embedding_path": dummy_elmo_embedding_path,
        "preprocessed_data_path": preprocessed_data_path,
        "only_last_turn": 0,
        "reverse_slot_order": 0,
    }
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "graph_lr": graph_lr,
    "adam_epsilon": adam_epsilon,
    "load_epoch":-1,
    "save_interval":save_interval,
    "load_model_path": "",
    "additional": {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "gradient_clipping": gradient_clipping,
    }
  },
  "valid": {
    "batch_size":valid_batch_size,
    "valid_gen_batch_size": valid_gen_batch_size,
    "num_valid_generation": 1000,
    "last_turn_only_generation": 1,
    "step_size": 1,
  },
  "test": {
    "evaluation_name": "test_evaluation",
    "load_epoch": -1,
    "batch_size": test_batch_size,
    "num_evaluation": 0,
    "load_model_path": "",
    "seed": 42,
    "plot_img": 1,
    "additional": {
        "multiprocessing": 4,
        "generate_data": 0,
    }
  }
}