import os
import torch
from utils.log_system import logger

class BaseExecutor():
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

    def save_checkpoint(self, epoch, batch_id=0, record_best_model=False):
        state = {
            'epoch': epoch,
            'batch_id': batch_id,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        if record_best_model:
            file_name = "model_best.pth.tar"
            path_save_model = os.path.join(self.config.saved_model_path, file_name)
            torch.save(state, path_save_model)
            logger.print('Model Saved:', path_save_model)
        else:
            file_name = "model_{}.pth.tar".format(epoch)
            path_save_model = os.path.join(self.config.saved_model_path, file_name)
            # Save the state
            torch.save(state, path_save_model)
            logger.print('Model Saved:', path_save_model)

            file_name = "model_lastest.pth.tar".format(epoch)
            path_save_model = os.path.join(self.config.saved_model_path, file_name)
            # Save the state
            torch.save(state, path_save_model)
            logger.print('Lastest Model Saved:', path_save_model)

    def load_checkpoint_model(self, load_epoch=-1, load_best_model=False, load_model_path=""):
        if load_model_path:
            path_save_model = load_model_path
        else:
            if load_best_model:
                file_name = "model_best.pth.tar"
            else:
                if load_epoch == -1:
                    file_name = "model_lastest.pth.tar"
                else:
                    file_name = "model_{}.pth.tar".format(load_epoch)

            path_save_model = os.path.join(self.config.saved_model_path, file_name)

        try:
            logger.print("Loading checkpoint '{}'".format(path_save_model))
            # checkpoint = torch.load(filename)
            checkpoint = torch.load(path_save_model, map_location='cuda:{}'.format(self.config.gpu_device))
            self.loaded_epoch = int(checkpoint['epoch'])
            if not load_model_path:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.load_pretrain_weights(checkpoint['state_dict'])
            if not load_model_path:
                # Loading from external model weights, do not load optimizer
                if 'optimizer' in checkpoint.keys():
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    print('< optimizer loaded from checkpoint >')
                if 'scheduler' in checkpoint.keys():
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    print('< scheduler loaded from checkpoint >')
            if 'batch_id' in checkpoint.keys():
                batch_id = checkpoint['batch_id']
            else:
                batch_id = 0
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Checkpoint loaded successfully from '{}' at (epoch {} batch {})\n"
                  .format(path_save_model, checkpoint['epoch'], batch_id))

            if load_model_path:
                self.loaded_epoch = 0 # Load model == start from epoch 0
        except OSError as e:
            self.loaded_epoch = 0
            print(e)
            print("No checkpoint exists from '{}'. Skipping...".format(path_save_model))
            print("**First time to train**")

    def train(self):
        raise NotImplementedError("Train function has not been defined!")


    def load_pretrain_weights(self, pretrained_dict):
        model_dict = self.model.state_dict()
        # 1. filter out unnecessary keys
        ignored_keys = []
        if len(self.config.ignore_pretrained_weights) > 0:
            for key in pretrained_dict.keys():
                for ignore_str in self.config.ignore_pretrained_weights:
                    if ignore_str in key:
                        ignored_keys.append(key)
        print('follwing pretrained weights are ignored', ignored_keys)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and k not in ignored_keys}
        print('Loading pretrained weights', [k for k in pretrained_dict.keys()])
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)