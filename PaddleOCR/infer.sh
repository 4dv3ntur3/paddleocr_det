image_dir="./infer_src"
image_dir_small="./infer_src_small"
det_model_dir=$1

unclip_ratio=0.8
box_thresh=0.6
# infer_result=$2

draw_img_save_dir="$1+_+$2+_small_text"

# python3 tools/infer/predict_det.py --image_dir=${image_dir} --det_model_dir=$1 --det_db_unclip_ratio=${unclip_ratio} --det_db_box_thresh=${box_thresh}

python3 tools/infer/predict_det.py --image_dir=${image_dir_small} --det_model_dir=$1 --draw_img_save_dir=${draw_img_save_dir} --det_db_box_thresh=0.1  --det_db_unclip_ratio=$2 --det_db_score_mode="fast"

# python3 tools/infer/predict_det.py --image_dir=${image_dir} --det_model_dir=$1 --draw_img_save_dir=$2


# sample
# sh infer.sh ./v3_det_student_crawl_word_0.6_best_33 ./