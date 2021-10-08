rm res
test_file=$1
ALLENNLP_PATH=/home/pc/.virtualenvs/Convlab/bin/allennlp
for i in 43; do
python ${ALLENNLP_PATH} predict --cuda-device 0 --predictor dstqa --include-package dstqa --weights-file model/model_state_epoch_${i}.th model/model.tar.gz ${test_file} > ${i}
python formulate_pred_belief_state.py ${i} >> res
done
