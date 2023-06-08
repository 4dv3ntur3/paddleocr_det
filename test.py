import cv2




origin = cv2.imread('./infer_src/scan_test_1.jpg')

origin_cvt = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

print(origin_cvt.shape)
quit()

# origin_cvt_again = cv2.cvtColor(origin, cv2.COLOR_GRAY2BGR)




origin = cv2.imread('./infer_src/scan_test_1.jpg', cv2.IMREAD_GRAYSCALE)

# origin_cvt = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
origin_cvt_again = cv2.cvtColor(origin, cv2.COLOR_GRAY2BGR)

# 이렇게 하면 3 channel을 유지하면서 rgb를 할 수 있기는 함... 
# 근데 model이 3 channel image만 학습하도록 했는데... normalize해도 잘될까?...



print(origin.shape)
# print(origin_cvt.shape)
print(origin_cvt_again.shape)


for i in range(origin_cvt_again.shape[2]):
    
    print(origin_cvt_again[i])




# cv2.imwrite('./output_test.jpg', origin_cvt_again)


