import os
import albumentations as A
from collections import defaultdict
from PIL import Image, ImageOps
import cv2
import numpy as np
import torch
import random
import math
import uuid
from pathlib import Path
import json
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
from openpyxl.utils import get_column_letter
import shutil
from io import BytesIO

import src.config as config
from src.recognizer import load_model_CRNN2
from src.converter import StrLabelConverter
from src.barcode_decoder import try_different_parameters


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def preprocess(img, transform=None):
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
        x = cv2.copyMakeBorder(x, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))

    x = x.astype(np.float32) / 255.
    x = x.transpose(2, 0, 1)
    x = torch.tensor(x)

    return x


def is_price_tag_below_product(product_bbox, price_tag, tolerance_multiplier=0.8):
    product_x_left = product_bbox[0]
    product_x_right = product_bbox[0] + product_bbox[2]
    product_y_top = product_bbox[1]
    product_y_bottom = product_bbox[1] + product_bbox[3]

    price_tag_x_left = price_tag[0]
    price_tag_x_right = price_tag[0] + price_tag[2]
    price_tag_y_top = price_tag[1]
    price_tag_y_bottom = price_tag[1] + price_tag[3]

    vertical_tolerance = price_tag[2] * tolerance_multiplier

    is_vertically_aligned = (product_y_bottom <= price_tag_y_top <= product_y_bottom + vertical_tolerance) or \
                            (product_y_top < price_tag_y_top <= product_y_bottom)

    is_horizontally_overlapping = not (price_tag_x_right < product_x_left or price_tag_x_left > product_x_right)

    return is_vertically_aligned and is_horizontally_overlapping


def find_closest_product_above_price_tag(price_tag, products, tolerance_multiplier=0.8):
    price_tag_center = (price_tag[0] + price_tag[2] / 2, price_tag[1] + price_tag[3] / 2)
    potential_matches = []

    for product in products:
        product_bbox = product['bbox']
        product_center = (product_bbox[0] + product_bbox[2] / 2, product_bbox[1] + product_bbox[3] / 2)

        if is_price_tag_below_product(product_bbox, price_tag, tolerance_multiplier):
            distance = ((product_center[0] - price_tag_center[0]) ** 2 + (product_center[1] - price_tag_center[1]) ** 2) ** 0.5
            potential_matches.append((product, distance))

    if potential_matches:
        closest_product, _ = min(potential_matches, key=lambda x: x[1])
        return closest_product['id'], closest_product
    return None, None


def determine_final_matches(annotations, unmatched_price_tags_list):
    product_to_price_tags = defaultdict(list)
    matched_products = set()

    for price_tag in unmatched_price_tags_list:
        product_id, closest_product = find_closest_product_above_price_tag(price_tag, annotations)
        if closest_product:
            product_key = tuple(closest_product['bbox'])
            price_tag_info = {'bbox': tuple(price_tag), 'recognized_text': None, 'product_id': product_id}
            product_to_price_tags[product_key].append(price_tag_info)
            matched_products.add(product_key)

    final_matches = {}
    for product_bbox, matches in product_to_price_tags.items():
        final_matches[product_bbox] = matches[0]

    unmatched_price_tags = [pt for pt in unmatched_price_tags_list if tuple(pt) not in [match['bbox'] for match in final_matches.values()]]

    for price_tag in unmatched_price_tags:
        for annotation in annotations:
            product_key = tuple(annotation['bbox'])
            if product_key not in matched_products and is_price_tag_below_product(annotation['bbox'], price_tag):
                final_matches[product_key] = {'bbox': tuple(price_tag), 'recognized_text': None, 'product_id': annotation['id']}
                matched_products.add(product_key)
                break

    matched_price_tags = [info['bbox'] for info in final_matches.values()]
    unmatched_price_tags_final = [pt for pt in unmatched_price_tags if tuple(pt) not in matched_price_tags]
    unmatched_products_final = [product for product in annotations if tuple(product['bbox']) not in matched_products]

    return final_matches, unmatched_price_tags_final, unmatched_products_final


def extract_price_tag_image(full_image, bbox):
    x, y, w, h = bbox
    price_tag_image = full_image[y:y+h, x:x+w]
    return price_tag_image


def recognize_text_on_price_tag(price_tag_image, model_rubles, converter, transformer):
    preprocessed_image = preprocess(price_tag_image, transform=transformer).unsqueeze(0)
    cuda_image = preprocessed_image.cuda()

    predictions_rubles, cls_rubles = model_rubles(cuda_image)
    cls_idx_rubles = cls_rubles.argmax(1).item()
    predictions_rubles = predictions_rubles.permute(1, 0, 2).contiguous()
    prediction_size_rubles = torch.IntTensor([predictions_rubles.size(0)]).repeat(1)
    _, predicted_labels_rubles = predictions_rubles.detach().cpu().max(2)
    predicted_test_labels = np.array(converter.decode(predicted_labels_rubles, prediction_size_rubles, raw=False))

    return str(predicted_test_labels), config.regions[cls_idx_rubles]


def draw_bboxes_with_dynamic_colors_and_lines(image_path, final_matches, unmatched_price_tags, unmatched_products,
                                              image_output_path, font_scale=config.font_scale, thickness=config.thickness):
    image = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for product_bbox, tag_data in final_matches.items():
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        bbox = tag_data['bbox']
        recognized_text = tag_data['recognized_text']
        promo = tag_data['promo']

        cv2.rectangle(image, (product_bbox[0], product_bbox[1]),
                      (product_bbox[0] + product_bbox[2], product_bbox[1] + product_bbox[3]), color, thickness)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, thickness)

        line_start = (product_bbox[0] + product_bbox[2] // 2, product_bbox[1] + product_bbox[3])
        line_end = (bbox[0] + bbox[2] // 2, bbox[1])
        cv2.line(image, line_start, line_end, color, thickness)

        line_length = math.sqrt((line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2)
        line_length_text = f"{line_length:.1f}px"
        mid_point = ((line_start[0] + line_end[0]) // 2, (line_start[1] + line_end[1]) // 2)
        cv2.putText(image, line_length_text, mid_point, font, font_scale, color, thickness)

        if recognized_text:
            text_size = cv2.getTextSize(recognized_text, font, font_scale, thickness)[0]
            text_width, text_height = text_size

            text_x = bbox[0]
            text_y = bbox[1] + bbox[3] + text_height + 10
            if promo == 'promo':
                cv2.rectangle(image, (text_x, text_y + 5), (text_x + text_width, text_y - text_height - 10), (0, 0, 255),
                              cv2.FILLED)
            else:
                cv2.rectangle(image, (text_x, text_y + 5), (text_x + text_width, text_y - text_height - 10), (0, 0, 0),
                              cv2.FILLED)

            cv2.putText(image, recognized_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    for product in unmatched_products:
        bbox = product['bbox']
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), config.grey_color, thickness)

    for price_tag in unmatched_price_tags:
        cv2.rectangle(image, (price_tag[0], price_tag[1]), (price_tag[0] + price_tag[2], price_tag[1] + price_tag[3]),
                      config.grey_color, thickness)

    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path, exist_ok=True)
    cv2.imwrite(os.path.join(image_output_path, os.path.basename(image_path)), image)


def save_crop(image_path_full, bbox, output_folder, prefix=""):
    crop_path = f"{output_folder}/{prefix}_{uuid.uuid4()}.jpg"
    crop_image = extract_price_tag_image(cv2.imread(str(image_path_full)), bbox)
    cv2.imwrite(crop_path, crop_image)
    return crop_path


def calculate_line_length(bbox1, bbox2):
    center1 = (bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2)
    center2 = (bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2)
    return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def expand_bbox(bbox, padding=config.padding, image_shape=None):
    x, y, w, h = bbox
    x_expanded = max(x - padding, 0)
    y_expanded = max(y - padding, 0)
    if image_shape:
        w_expanded = min(w + 2 * padding, image_shape[1] - x_expanded)
        h_expanded = min(h + 2 * padding, image_shape[0] - y_expanded)
    else:
        w_expanded = w + 2 * padding
        h_expanded = h + 2 * padding

    return (x_expanded, y_expanded, w_expanded, h_expanded)


def extract_image_data(image, bbox, format):
    price_tag_image = extract_price_tag_image(image, bbox)
    is_success, buffer = cv2.imencode('.jpg', price_tag_image)
    if not is_success:
        pil_image = Image.fromarray(cv2.cvtColor(price_tag_image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        pil_image.save(buffered, format=format)
        return buffered.getvalue()
    return buffer.tobytes()


def collect_data(image_info, matched, unmatched_price_tags_final, unmatched_products_final, images_path, output_folder, model_rubles, converter, transformer, with_unmatched = False):
    collected_data = []
    image_name = image_info['file_name']
    image_path_full = Path(images_path) / image_name
    image_id = image_info['id']

    full_image = cv2.imread(str(image_path_full))
    image_shape = full_image.shape[:2]

    image_format = image_path_full.suffix.lower().replace('.', '').upper()

    for product_bbox, price_tag_data in matched.items():
        # sku_crop_path = save_crop(image_path_full, product_bbox, output_folder, prefix="SKU")

        expanded_bbox = expand_bbox(price_tag_data['bbox'], padding=5, image_shape=image_shape)
        # price_tag_crop_path = save_crop(image_path_full, expanded_bbox, output_folder, prefix="PriceTag")

        line_start = (product_bbox[0] + product_bbox[2] // 2, product_bbox[1] + product_bbox[3])
        line_end = (price_tag_data['bbox'][0] + price_tag_data['bbox'][2] // 2, price_tag_data['bbox'][1])

        recognized_text, promo = recognize_text_on_price_tag(
            extract_price_tag_image(cv2.imread(str(image_path_full)), price_tag_data['bbox']),
            model_rubles, converter, transformer
        )
        match_length = calculate_line_length(product_bbox, expanded_bbox)

        image_data = extract_image_data(full_image, price_tag_data['bbox'], format=image_format)
        decoded_barcodes, decode_params, _ = try_different_parameters(image_data)
        barcode_result = decoded_barcodes if decoded_barcodes else ''

        data_entry = {
            'sku_id': price_tag_data['product_id'],
            'img_name': image_name,
            'img_id': image_id,
            'price_bbox': price_tag_data['bbox'],
            'match_points': [(line_start[0], line_start[1]), (line_end[0], line_end[1])],
            # 'sku': sku_crop_path,
            # 'price_tag': price_tag_crop_path,
            'match_length': f"{match_length:.1f}px",
            'promo': promo,
            'sku_price': recognized_text,
            'barcode': barcode_result,
        }
        collected_data.append(data_entry)

    if with_unmatched is True:
        for product in unmatched_products_final:
            if product['image_id'] == image_id:
                # sku_crop_path = save_crop(image_path_full, product['bbox'], output_folder, prefix="SKU")
                data_entry = {
                    'img_name': image_name,
                    # 'sku': sku_crop_path,
                    'price_tag': '',
                    'match_length': '',
                    'promo': '',
                    'sku_price': '',
                    'price_check': '',
                    'promo_check': '',
                    'match_correct': ''
                }
                collected_data.append(data_entry)

        for price_tag in unmatched_price_tags_final:
            expanded_bbox = expand_bbox(price_tag, padding=5, image_shape=image_shape)
            # price_tag_crop_path = save_crop(image_path_full, expanded_bbox, output_folder, prefix="PriceTag")
            recognized_text, promo = recognize_text_on_price_tag(
                extract_price_tag_image(cv2.imread(str(image_path_full)), price_tag),
                model_rubles, converter, transformer
            )
            data_entry = {
                'img_name': image_name,
                'sku': '',
                # 'price_tag': price_tag_crop_path,
                'match_length': '',
                'promo': promo,
                'sku_price': recognized_text,
                'price_check': '',
                'promo_check': '',
                'match_correct': ''
            }
            collected_data.append(data_entry)

    return collected_data


def is_price_tag_near_edge(bbox, image_shape, edge_tolerance=config.edge_tolerance):
    x, y, w, h = bbox
    image_width, image_height = image_shape

    return (x <= edge_tolerance or y <= edge_tolerance or
            (x + w) >= (image_width - edge_tolerance) or
            (y + h) >= (image_height - edge_tolerance))


def recognize_crossed_price_tag_text(price_tag_image, recognizer, converter, transformer):
    preprocessed_image = preprocess(price_tag_image, transform=transformer).unsqueeze(0)
    cuda_image = preprocessed_image.cuda()

    predictions, cls = recognizer(cuda_image)
    cls_idx = cls.argmax(1).item()
    predictions = predictions.permute(1, 0, 2).contiguous()
    prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
    _, predicted_labels = predictions.detach().cpu().max(2)
    predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))

    return str(predicted_test_labels), config.regions[cls_idx]


def detect_crossed_price_tags(detector_results):
    crossed_price_tags = [result['bbox'] for result in detector_results['annotations']]
    return crossed_price_tags


def process_data(sku_data_path, results_price, images_path, model_path, crossed_detector, crossed_recognizer):
    with open(sku_data_path, 'r') as file:
        sku_data = json.load(file)

    model_rubles = load_model_CRNN2(model_path)
    transformer = A.Compose([A.NoOp()])
    converter = StrLabelConverter(config.alphabet)

    images_info = sku_data['job']['result']['annotation']['images']
    categories = sku_data['job']['result']['annotation']['categories']
    sku_category_ids = [category['id'] for category in categories if category['supercategory'] == 'sku']
    collected_data = []

    for image_info in images_info:
        image_id = image_info['id']
        processing_image_name = image_info['file_name']
        image_path = os.path.join(images_path, processing_image_name)
        image_shape = (image_info['width'], image_info['height'])

        filtered_price_tags = [tag for tag in results_price['annotations'] if tag['image_id'] == image_id]
        unmatched_price_tags_list = [tag['bbox'] for tag in filtered_price_tags]
        unmatched_price_tags_list = [tag for tag in unmatched_price_tags_list if
                                     not is_price_tag_near_edge(tag, image_shape)]

        annotations_filtered = [annotation for annotation in sku_data['job']['result']['annotation']['annotations'] if
                                annotation['category_id'] in sku_category_ids and annotation['image_id'] == image_id]

        matched, unmatched_price_tags_final, unmatched_products_final = determine_final_matches(annotations_filtered,
                                                                                                unmatched_price_tags_list)

        if crossed_detector['annotations']:
            crossed_price_tags = detect_crossed_price_tags(crossed_detector)
            for tag in crossed_price_tags:
                recognized_text, promo = recognize_crossed_price_tag_text(extract_price_tag_image(cv2.imread(image_path), tag), crossed_recognizer, converter, transformer)
                matched[tuple(tag)] = {'bbox': tag, 'recognized_text': recognized_text, 'promo': promo, 'product_id': None, 'crossed': True}


        collected_data += collect_data(image_info, matched, unmatched_price_tags_final, unmatched_products_final,
                                       images_path, f'{images_path}crops_excel', model_rubles,
                                       converter, transformer, with_unmatched=False)

    return collected_data


def update_annotations_with_new_data(original_data, new_data):
    annotation_dict = {(item['id'], item['image_id']): item for item in
                       original_data['job']['result']['annotation']['annotations']}

    for data in new_data:
        key = (data['id'], data['image_id'])
        if key in annotation_dict:
            annotation_dict[key].update(data)

    updated_annotations = list(annotation_dict.values())
    original_data['job']['result']['annotation']['annotations'] = updated_annotations
    return original_data


def integrate_data(collected_data, sku_annotation_path, images_folder):
    with open(sku_annotation_path, 'r') as file:
        annotations_data = json.load(file)

    annotations_by_image_id = {}
    for annotation in annotations_data['job']['result']['annotation']['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(annotation)

    integrated_results = []
    for result in collected_data:
        img_id = result['img_id']
        sku_id = result['sku_id']
        if img_id in annotations_by_image_id:
            annotations = annotations_by_image_id[img_id]
            for annotation in annotations:
                if annotation['id'] == sku_id:
                    integrated_result = {
                        **annotation,
                        'price_bbox': result['price_bbox'],
                        'match_points': result['match_points'],
                        'match_length': result['match_length'],
                        'promo': result['promo'],
                        'sku_price': result['sku_price'],
                        'barcode': result['barcode'],
                    }
                    integrated_results.append(integrated_result)
                    break

    updated_annotations = update_annotations_with_new_data(annotations_data, integrated_results)

    with open(f'{images_folder}/integrated_results.json', 'w') as f:
        json.dump(updated_annotations, f, indent=4)
