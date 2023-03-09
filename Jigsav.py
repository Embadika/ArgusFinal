import cv2
import numpy as np
copy = cv2.imread('C:/Users/zyrik/Pictures/argus/(3).png')
deep_copy = copy.copy()  # .copy()
# output = cv2.resize(deep_copy, dsize)
# cv2.imshow('result', output)
# cv2.waitKey(0)
thresh = cv2.cvtColor(deep_copy, cv2.COLOR_BGR2RGB)
# формируем начальный и конечный цвет фильтра

# h_min = np.array((r1, g1, b1), np.uint8)
# h_max = np.array((r2, g2, b2), np.uint8)
# thresh = cv2.inRange(thresh, h_min, h_max)
thresh = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
thresh = cv2.medianBlur(thresh, 1 + 2 * 2)
ret, thresh = cv2.threshold(thresh, 150, 255, cv2.THRESH_BINARY)
thresh = 255 - thresh
thresh = cv2.bitwise_not(thresh)
# output = cv2.resize(thresh, dsize)
# # # print(r1, g1, b1, r2, g2, b2, bl, tr1, tr2)
cv2.imshow('result', thresh)
cv2.waitKey(0)
shapes, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
findjig = []
for i, cnt in enumerate(shapes):
    x, y, w, h = cv2.boundingRect(cnt)
    lenght = cv2.arcLength(cnt, True)
    M = cv2.moments(cnt)
    findjig.append({"x":x, "y":y, "w":w,"h":h,"len":lenght,"m":M})
    print (lenght, M)
    # print(x, y, w, h)
    if x == 0 and y == 0 and h == deep_copy.shape[1] and w == deep_copy.shape[0]:
        continue
    if w < 15 or h < 15:
        continue
    if h * w < 150:
        continue
    else:
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 10)

from operator import itemgetter
findjig = sorted(findjig, key=itemgetter('len'))

for i, f in enumerate(findjig):
    # text = ""
    # if i == 0:
    #     text="5"
    # elif i ==1:
    #     text = "5"
    cv2.putText(copy, str(5-i), (f["x"], f["y"]), cv2.FONT_HERSHEY_SIMPLEX, 4, 255)

cv2.imshow("res", copy)
cv2.waitKey(0)
