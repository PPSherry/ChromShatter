import cv2

def get_coordinates(image_path):
    img = cv2.imread(image_path)
    cv2.imshow("Image", img)

    # 鼠标点击事件
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Coordinate: ({x}, {y})")

    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例使用
image_path = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/pdf_convert/page_1.png"
get_coordinates(image_path)