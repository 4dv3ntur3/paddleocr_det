import glob
import shutil, os



with open('model_dir/infer_sample.txt', 'r') as f:
    file_list = f.readlines()
    

root = "model_dir/v3_det_student_crawl_word_0.6_best_50_unclip_ratio_1.3_normal_box_thresh_0.6_none_erode_hard"

root_mask = "model_dir/v3_det_student_crawl_word_0.6_best_50_unclip_ratio_1.3_normal_box_thresh_0.6_none_erode_hard/mask"

dest_dir = "model_dir/v3_det_student_crawl_word_0.6_best_50_unclip_ratio_1.3_normal_box_thresh_0.6_none_erode_hard/infer_check"

# img_list = glob.glob(os.path.abspath(root)+"**/*.jpg")
# mask_list = glob.glob(os.path.abspath(root_mask)+"**/*.jpg")


counter = 1
for file in file_list:
    
    fn = file.strip("\n")
    fn = fn.split(os.sep)[-1]
    
    fn_fn = os.path.splitext(fn)[0] +"_mask" + os.path.splitext(fn)[1]
    
    img_check_fn = os.path.join(os.path.abspath(root), fn)
    mask_check_fn = os.path.join(os.path.abspath(root_mask), fn)
    
    if os.path.isfile(img_check_fn):
        counter += 1
        
        new_fn = os.path.join(dest_dir, img_check_fn)
        shutil.move(img_check_fn, new_fn)
    
    if os.path.isfile(mask_check_fn):
        counter += 1
        new_fn_mask = os.path.join(dest_dir, fn_fn)
        shutil.move(mask_check_fn, new_fn_mask)
