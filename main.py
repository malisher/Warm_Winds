import argparse

from src.utils import parse_annotations, process_images
from src.excel_generator import generate_excel

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='data/792_annotations/', help='Folder containing all data')
    parser.add_argument('--annotations_path', default='data/792_annotations/annotations.xml', help='Path to annotation')
    parser.add_argument('--brands_path', default='', help='Path to filtered brands excel file')
    parser.add_argument('--process_by_brands', default=False, action="store_true", help='True if process brands, else leave blank (if SKU)')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = parse_opt()

    images_data = parse_annotations(args.annotations_path)
    processed_images_data, cover_all = process_images(images_data, args.brands_path, args.data_folder, args.process_by_brands)

    generate_excel(processed_images_data, args.data_folder, cover_all)
