import argparse
import os
import cv2
import torch
import albumentations as A
from PIL import Image, ImageOps
import numpy as np
import uuid
from pathlib import Path
from src.detector import run_detector
from src.recognizer import load_model_CRNN as load_crossed_recognizer
from src.recognizer import load_model_CRNN2
from src.converter import StrLabelConverter
import src.config as config
import time
import logging
from multiprocessing import Pool, cpu_count
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
multiprocessing.set_start_method('spawn', force=True)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', default='data/test_input', help='Folder containing images')
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
    parser.add_argument('--output_folder', default='data/test_output', help='Folder to save cropped images')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of images to process simultaneously')

    opt = parser.parse_args()
    return opt


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def preprocess(img, transform=None):
    if img is None or img.size == 0:
        raise ValueError("The input image is empty.")

    original_image = img.copy()
    x = cv2.resize(original_image, (config.img_w, config.img_h))
    if transform is not None:
        x = transform(image=x)['image']

    h, w, _ = img.shape
    left = 0
    right = 0
    top = 0
    bottom = 0
    if w > h:
        h = int(h * config.img_w / w)
        w = config.img_w
        top = (config.img_h - h) // 2
        bottom = config.img_h - h - top
    else:
        w = int(w * config.img_h / h)
        h = config.img_h
        left = (config.img_w - w) // 2
        right = config.img_w - w - left
    if top < 0 or bottom < 0 or left < 0 or right < 0:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        im_pil = Image.fromarray(img)
        img = resize_with_padding(im_pil, (config.img_w, config.img_h))
        x = np.asarray(img)
        if transform is not None:
            x = transform(image=x)['image']
    else:
        x = cv2.resize(img, (w, h))
        if transform is not None:
            x = transform(image=x)['image']
        x = cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    x = x.astype(np.float32) / 255.
    x = x.transpose(2, 0, 1)
    x = torch.tensor(x)

    return x


def recognize_text_on_price_tag(price_tag_image, model, converter, transformer, is_crossed=False):
    if price_tag_image is None or price_tag_image.size == 0:
        raise ValueError("The price tag image is empty.")

    preprocessed_image = preprocess(price_tag_image, transform=transformer).unsqueeze(0)
    cuda_image = preprocessed_image.cuda()

    if is_crossed:
        predictions = model(cuda_image)
        predictions = predictions.permute(1, 0, 2).contiguous()
        prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
        predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
        predicted_probs = np.around(torch.exp(predicted_probs).permute(1, 0).numpy(), decimals=1)
        predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
        return str(predicted_test_labels)
    else:
        predictions, cls = model(cuda_image)
        cls_idx = cls.argmax(1).item()
        predictions = predictions.permute(1, 0, 2).contiguous()
        prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
        _, predicted_labels = predictions.detach().cpu().max(2)
        predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
        return str(predicted_test_labels), config.regions[cls_idx]


def extract_price_tag_image(full_image, bbox):
    x, y, w, h = bbox
    price_tag_image = full_image[y:y + h, x:x + w]
    return price_tag_image


def process_image(img_path, detector_results, regular_recognizer, crossed_recognizer, converter, transformer,
                  output_folder, crossed_detector_weight_path, detector_classes_path):
    try:
        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"Failed to read image: {img_path}")
            return

        img_file = os.path.basename(img_path)
        image_id = detector_results['images'].get(img_file)

        if image_id is None:
            logging.warning(f"No detection results for image: {img_file}")
            return

        for bbox in detector_results['annotations']:
            if bbox['image_id'] == image_id:
                cropped_img = extract_price_tag_image(image, bbox['bbox'])
                if cropped_img is None or cropped_img.size == 0:
                    logging.warning(f"Skipped empty image for bbox: {bbox['bbox']} in file: {img_file}")
                    continue

                try:
                    recognized_text, region = recognize_text_on_price_tag(cropped_img, regular_recognizer, converter,
                                                                          transformer)
                    cropped_img_path = output_folder / f"{image_id}_{recognized_text}_{uuid.uuid4()}.jpg"
                    cv2.imwrite(str(cropped_img_path), cropped_img)

                    # Process for crossed-out price tags
                    crossed_results = run_detector(str(cropped_img_path), crossed_detector_weight_path,
                                                   detector_classes_path)

                    for crossed_bbox in crossed_results['annotations']:
                        crossed_cropped_img = extract_price_tag_image(cropped_img, crossed_bbox['bbox'])
                        if crossed_cropped_img is None or crossed_cropped_img.size == 0:
                            logging.warning(
                                f"Skipped empty crossed image for bbox: {crossed_bbox['bbox']} in file: {cropped_img_path}")
                            continue

                        crossed_recognized_text = recognize_text_on_price_tag(crossed_cropped_img, crossed_recognizer,
                                                                              converter, transformer, is_crossed=True)
                        new_output_path = output_folder / f"crossed_{image_id}_{crossed_recognized_text}_{uuid.uuid4()}.jpg"
                        cv2.imwrite(str(new_output_path), crossed_cropped_img)

                        os.remove(str(cropped_img_path))
                        break  # Assume only one crossed-out price tag per crop

                except Exception as e:
                    logging.error(f"Error processing cropped image: {e}")

    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")


def initialize_models(recognizer_weight_path, crossed_recognizer_weight_path):
    regular_recognizer = load_model_CRNN2(recognizer_weight_path)
    crossed_recognizer = load_crossed_recognizer(crossed_recognizer_weight_path)
    transformer = A.Compose([A.NoOp()])
    converter = StrLabelConverter(config.alphabet)
    return regular_recognizer, crossed_recognizer, transformer, converter


def process_image_batch(args):
    img_paths, detector_results, recognizer_weight_path, crossed_recognizer_weight_path, output_folder, crossed_detector_weight_path, detector_classes_path = args

    regular_recognizer, crossed_recognizer, transformer, converter = initialize_models(recognizer_weight_path,
                                                                                       crossed_recognizer_weight_path)

    for img_path in img_paths:
        process_image(img_path, detector_results, regular_recognizer, crossed_recognizer, converter, transformer,
                      output_folder, crossed_detector_weight_path, detector_classes_path)


def main():
    args = parse_opt()

    results_price = run_detector(args.images_folder, args.detector_weight_path, args.detector_classes_path)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png"}

    images_dict = {img['file_name']: img['id'] for img in results_price['images']}
    results_price['images'] = images_dict  # Update the results_price dictionary

    image_paths = [os.path.join(args.images_folder, img_file) for img_file in os.listdir(args.images_folder)
                   if Path(img_file).suffix.lower() in valid_extensions]

    batches = [image_paths[i:i + args.batch_size] for i in range(0, len(image_paths), args.batch_size)]

    process_args = [(batch, results_price, args.recognizer_weight_path, args.crossed_recognizer_weight_path,
                     output_folder, args.crossed_detector_weight_path, args.detector_classes_path) for batch in batches]

    with Pool(processes=min(cpu_count(), len(batches))) as pool:
        pool.map(process_image_batch, process_args)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
