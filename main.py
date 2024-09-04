# Script without multiprocessing
#
# import argparse
# from src.detector import run_detector
# from src.utils import process_data, integrate_data
# from src.recognizer import load_model_CRNN as load_crossed_recognizer
# import time
#
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--images_folder', default='data/panorama', help='Folder containing images')
#     parser.add_argument('--sku_annotation_path', default='data/panorama/2819011.json',
#                         help='Annotation json path including file name and extension')
#     parser.add_argument('--detector_weight_path', default='weights/best290424x.pt',
#                         help='Detector model path including weight name and extension')
#     parser.add_argument('--detector_classes_path', default='weights/classes.pkl',
#                         help='Detector classes path including file name and extension (.pkl)')
#     parser.add_argument('--recognizer_weight_path', default='weights/99_500_Train:_22.0904,_Accuracy:_0.9343,_Val:_21.1236,_Accuracy:_0.9464,_Cls_Accuracy:_0.9286,_lr:_1e-05.pth',
#                         help='Recognizer model path including weight name and extension')
#     parser.add_argument('--crossed_detector_weight_path', default='weights/crossed_best250624x.pt', help='Crossed-out detector model path including weight name and extension')
#     parser.add_argument('--crossed_recognizer_weight_path', default='weights/crossed_99_500_Train__5_0025,_Accuracy__0_8613,_Val__4_0938,_Accuracy__0.pth', help='Crossed-out recognizer model path including weight name and extension')
#
#     opt = parser.parse_args()
#     return opt
#
#
# if __name__ == '__main__':
#     args = parse_opt()
#     results_price = run_detector(args.images_folder, args.detector_weight_path, args.detector_classes_path)
#     crossed_recognizer = load_crossed_recognizer(args.crossed_recognizer_weight_path)
#     crossed_detector = run_detector(args.images_folder, args.crossed_detector_weight_path, args.detector_classes_path)
#     collected_data = process_data(args.sku_annotation_path, results_price, args.images_folder, args.recognizer_weight_path, crossed_detector, crossed_recognizer)
#     integrate_data(collected_data, args.sku_annotation_path, args.images_folder)


import argparse
from src.utils import process_data, integrate_data
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging
import multiprocessing
import torch.backends.cudnn as cudnn


cudnn.enabled = False
multiprocessing.set_start_method('spawn', force=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', default='data/images/new_photos/1/2', help='Folder containing images')
    parser.add_argument('--sku_annotation_path', default='data/images/new_photos/1/2/2810113.json',
                        help='Annotation json path including file name and extension')
    parser.add_argument('--detector_weight_path', default='weights/best290424x.pt',
                        help='Detector model path including weight name and extension')
    parser.add_argument('--detector_classes_path', default='weights/classes.pkl',
                        help='Detector classes path including file name and extension (.pkl)')
    parser.add_argument('--recognizer_weight_path',
                        default='weights/99_500_Train:_22.0904,_Accuracy:_0.9343,_Val:_21.1236,_Accuracy:_0.9464,_Cls_Accuracy:_0.9286,_lr:_1e-05.pth',
                        help='Recognizer model path including weight name and extension')
    parser.add_argument('--crossed_detector_weight_path', default='weights/crossed_best250624x.pt',
                        help='Crossed-out detector model path including weight name and extension')
    parser.add_argument('--crossed_recognizer_weight_path',
                        default='weights/crossed_99_500_Train__5_0025,_Accuracy__0_8613,_Val__4_0938,_Accuracy__0.pth',
                        help='Crossed-out recognizer model path including weight name and extension')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of images to process simultaneously')

    opt = parser.parse_args()
    return opt


def process_image_batch(args):
    images_batch, args_dict = args
    from src.detector import run_detector
    from src.recognizer import load_model_CRNN as load_crossed_recognizer

    results_price = run_detector(images_batch, args_dict['detector_weight_path'], args_dict['detector_classes_path'])
    crossed_recognizer = load_crossed_recognizer(args_dict['crossed_recognizer_weight_path'])
    crossed_detector = run_detector(images_batch, args_dict['crossed_detector_weight_path'],
                                    args_dict['detector_classes_path'])

    collected_data = process_data(args_dict['sku_annotation_path'], results_price, args_dict['images_folder'],
                                  args_dict['recognizer_weight_path'], crossed_detector, crossed_recognizer)
    return collected_data


def main():
    args = parse_opt()

    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = [os.path.join(args.images_folder, img_file) for img_file in os.listdir(args.images_folder)
                   if Path(img_file).suffix.lower() in valid_extensions]

    batches = [image_paths[i:i + args.batch_size] for i in range(0, len(image_paths), args.batch_size)]

    args_dict = vars(args)
    process_args = [(batch, args_dict) for batch in batches]

    with Pool(processes=min(cpu_count(), len(batches))) as pool:
        results = pool.map(process_image_batch, process_args)

    all_collected_data = [item for sublist in results for item in sublist]

    integrate_data(all_collected_data, args.sku_annotation_path, args.images_folder)


if __name__ == '__main__':
    main()
