# recommended paddle.__version__ == 2.0.0
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml

python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o Global.pretrained_model=Multilingual_PP-OCRv3_det_distill_train/student Optimizer.base_lr=0.0001