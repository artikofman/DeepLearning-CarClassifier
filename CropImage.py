import cv2


ref_point = []
orig_image = None
processed_image = None
mouse_moving = 0


def shape_selection(event, x, y, flags, param):
    global ref_point, orig_image, processed_image, mouse_moving

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        mouse_moving = 1
        cv2.imshow("image", orig_image)
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_moving == 1:
            ref_point.append((x, y))
            mouse_moving = 2
        elif mouse_moving == 2:
            ref_point[1] = (x, y)
        if mouse_moving == 1 or mouse_moving == 2:
            processed_image = orig_image.copy()
            cv2.rectangle(processed_image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            cv2.imshow("image", processed_image)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_moving = 0


def main():
    global orig_image, processed_image
    orig_image = cv2.imread("C:\\Users\\orik\\Tests\\1.jpg")  # Kedem.jpeg")  # image0005.png")
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", shape_selection)

    # keep looping until the 'Esc' key is pressed
    while True:
        cv2.imshow("image", (orig_image if processed_image is None else processed_image))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):  # press 'r' to reset the window
            processed_image = None
        elif key == 27:
            break

    """
    if len(ref_point) == 2:
        try:
            crop_img = orig_image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            cv2.imshow("crop_img", crop_img)
            cv2.waitKey(0)
        except Exception as e:
            print(e)
    """

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
