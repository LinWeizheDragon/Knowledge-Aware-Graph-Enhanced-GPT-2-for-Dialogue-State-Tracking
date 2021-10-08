import sys
from subprocess import call
import torch
from utils.log_system import logger

def print_cuda_statistics():

    logger.print('__Python VERSION:  {}'.format(sys.version))
    logger.print('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.print('__CUDA VERSION')
    call(["nvcc", "--version"])
    logger.print('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.print('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.print('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.print('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.print('Available devices  {}'.format(torch.cuda.device_count()))
    logger.print('Current cuda device  {}'.format(torch.cuda.current_device()))