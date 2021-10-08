SAVE_MODEL_PATH=./model
ALLENNLP_PATH=/home/pc/.virtualenvs/Convlab/bin/allennlp
rm -r ${SAVE_MODEL_PATH}
python ${ALLENNLP_PATH} train config_nosp.jsonnet -s ${SAVE_MODEL_PATH} --include-package dstqa
