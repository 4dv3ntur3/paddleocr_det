unclip_ratio=0.8
box_thresh=0.6
erode_kernel=4

# --det_db_unclip_ratio=${unclip_ratio}
# --det_db_box_thresh=${box_thresh}
# --erode_kernel=${erode_kernel}

image_dir="../infer_src/infer_src_small"
air_model="../model_dir/v3_det_student_crawl_word_0.6_best_50"
fine_model="../model_dir/v3_det_student_color_augment_finetuning_0.6_best_35"
pre_model="../model_dir/v3_det_student_color_augment_0.6_pretrain_best_115"

# change detetcion model here (air, fine, pre)
det_model_dir=$air_model

draw_img_temp_dir=$(echo ${image_dir} | cut --delimiter='/' --fields=3)
draw_img_save_dir=$det_model_dir/$draw_img_temp_dir

rec_model_dir="../model_dir/color_aug"
rec_char_dict_path="../model_dir/color_aug/rec_dict.txt"
# rec_char_dict_path="/home/ejpark/paddleocr_det/PaddleOCR/ppocr/utils/en_dict.txt"
vis_font_path="../model_dir/Dotum.ttf"


# 1. only detection
python3 tools/infer/predict_det.py --image_dir=${image_dir} --det_model_dir=${det_model_dir} --draw_img_save_dir=${draw_img_save_dir}


# 2. det & rec
# python3 tools/infer/predict_system.py --image_dir=${image_dir} --det_model_dir=${det_model_dir} --rec_model_dir=${rec_model_dir} --use_angle_cls=false --rec_char_dict_path=${rec_char_dict_path} --draw_image_save_dir=${draw_image_save_dir}

# 3. try multiple times
# for var in 0.8 1.0 1.2 1.3 1.4 1.5
# do
#     draw_img_save_dir="$1_unclip_ratio_${var}"
#     python3 tools/infer/predict_det.py --image_dir=${image_dir} --det_model_dir=$1 --draw_img_save_dir=${draw_img_save_dir} --det_db_unclip_ratio=${var}
# done