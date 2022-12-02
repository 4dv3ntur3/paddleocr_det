image_dir="./infer_src"
image_dir_small="./infer_src_small"
image_dir_hard="./infer_src_hard"


det_model_dir=$1
unclip_ratio=1.3
box_thresh=0.6
erode_kernel=None

# for var in 0.8 1.0 1.2 1.3 1.4 1.5
# do
#     draw_img_save_dir="$1_unclip_ratio_${var}"
#     python3 tools/infer/predict_det.py --image_dir=${image_dir} --det_model_dir=$1 --draw_img_save_dir=${draw_img_save_dir} --det_db_unclip_ratio=${var}
# done


draw_img_save_dir="$1_unclip_ratio_${unclip_ratio}_normal_box_thresh_${box_thresh}_erode_kernel_${erode_kernel}__hard"


python3 tools/infer/predict_det.py --image_dir=${image_dir_hard} --det_model_dir=$1 --draw_img_save_dir=${draw_img_save_dir} --det_db_unclip_ratio=${unclip_ratio} --det_db_box_thresh=${box_thresh} # s--erode_kernel=${erode_kernel}


# draw_img_save_dir="$1_unclip_ratio_${unclip_ratio}_normal_box_thresh_${box_thresh}_none_erode_hard"
# python3 tools/infer/predict_det.py --image_dir=${image_dir} --det_model_dir=$1 --draw_img_save_dir=${draw_img_save_dir} --det_db_unclip_ratio=${unclip_ratio} --det_db_box_thresh=${box_thresh}

# python3 tools/infer/predict_det.py --image_dir=${image_dir} --det_model_dir=$1 --draw_img_save_dir=$2

# sample
# sh infer.sh ./v3_det_student_crawl_word_0.6_best_33 ./