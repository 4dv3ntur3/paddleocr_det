image_dir="./infer_src"
image_dir_small="./infer_src_small"
det_model_dir=$1

unclip_ratio=1.4
box_thresh=0.6
# infer_result=$2

unclip_list=(0.8, 1.0, 1.2, 1.3, 1.4, 1.5)

for var in unclip_list
do
    python3 tools/infer/predict_det.py --image_dir=${image_dir} --det_model_dir=$1 --draw_img_save_dir=${draw_img_save_dir} --det_db_unclip_ratio=${unclip_ratio} --det_db_score_mode="fast"
done





draw_img_save_dir="$1_${unclip_ratio}"

# python3 tools/infer/predict_det.py --image_dir=${image_dir} --det_model_dir=$1 --det_db_unclip_ratio=${unclip_ratio} --det_db_box_thresh=${box_thresh}

python3 tools/infer/predict_det.py --image_dir=${image_dir} --det_model_dir=$1 --draw_img_save_dir=${draw_img_save_dir} --det_db_unclip_ratio=${unclip_ratio} --det_db_score_mode="fast"

# python3 tools/infer/predict_det.py --image_dir=${image_dir} --det_model_dir=$1 --draw_img_save_dir=$2


# sample
# sh infer.sh ./v3_det_student_crawl_word_0.6_best_33 ./