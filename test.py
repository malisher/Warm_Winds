import cv2
import sys

def stitch_images(img1_path, img2_path, output_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        sys.exit(1)

    stitcher = cv2.Stitcher_create()

    (status, stitched) = stitcher.stitch([img1, img2])

    if status != cv2.Stitcher_OK:
        sys.exit(2)

    cv2.imwrite(output_path, stitched)

if __name__ == '__main__':
    try:
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
        output_path = 'panorama.jpg'
        stitch_images(img1_path, img2_path, output_path)
    except IndexError:
        print("python script.py 'path/to/img1.jpg' 'path/to/img2.jpg'")
