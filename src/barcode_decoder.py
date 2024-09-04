from PIL import Image, ImageEnhance
from skimage.filters import threshold_local
from pyzbar.pyzbar import decode
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import numpy as np


def preprocess_image(image, contrast, block_size):
    grayscale = image.convert('L')
    contrast_enhancer = ImageEnhance.Contrast(grayscale)
    enhanced_image = contrast_enhancer.enhance(contrast)
    # enhanced_image = enhanced_image.resize(
    #     (int(enhanced_image.width * scale_factor), int(enhanced_image.height * scale_factor)), Image.Resampling.LANCZOS
    # )
    image_np = np.array(enhanced_image)
    adaptive_thresh = threshold_local(image_np, block_size, offset=10)
    thresholded = (image_np > adaptive_thresh).astype(np.uint8) * 255
    processed_image = Image.fromarray(thresholded)
    return processed_image


def decode_image_with_params(image_data, contrast, block):
    image = Image.open(BytesIO(image_data))
    processed_image = preprocess_image(image, contrast, block)
    barcodes = decode(processed_image)
    if barcodes:
        return [[barcode.data.decode('utf-8'), barcode.type] for barcode in barcodes], (contrast, block), image
    return [], None, image


def try_different_parameters(image_data):
    # scale_factors = [1.0, 1.5, 2.0]
    contrasts = [0.8, 1.2, 1.6]
    block_sizes = [51, 77, 101]
    with ThreadPoolExecutor() as executor:
        futures = []
        # for scale in scale_factors:
        for contrast in contrasts:
            for block in block_sizes:
                futures.append(executor.submit(decode_image_with_params, image_data, contrast, block))
        for future in as_completed(futures):
            result, params, image = future.result()
            if result:
                return result, params, image
    return [], None, image
