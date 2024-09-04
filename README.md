# SKU_price_tag_pipeline

## Installation

```
pip install -r requirements.txt
```

## Docker commands

```
docker build -f dockerfiles/Dockerfile -t price_matching .
```

## Example of usage

```
/opt/conda/bin/python3 main.py --images_folder=data/panorama/ --sku_annotation_path=data/panorama/2819011.json
```

## Possible arguments:

--images_folder - Folder containing images

--sku_annotation_path - Annotation json path including file name and extension

--detector_weight_path - Detector model path including weight name and extension

--detector_classes_path - Detector classes path including file name and extension (.pkl)

--recognizer_weight_path - Recognizer model path including weight name and extension

--excel_output_path - Excel output path including file name and extension

--crossed_detector_weight_path - Crossed-out detector model path including weight name and extension

--crossed_recognizer_weight_path - Crossed-out recognizer model path including weight name and extension

--batch_size - Number of images to process simultaneously

--draw_results (optional) - True if need to draw results (for debugging)

## Json output explanation

Only annotations part gets changed. If sku and price_tag matched:

price_bbox - bbox of price_tag

match_points - points of connection between sku and price_tag

match_length - connection length in pixels

promo - is price_tag promo or standard

sku_price - price recognition result

```
"annotations": 
[
...,
{
    "id": 1,
    "bbox": [
        392,
        577,
        59,
        73
    ],
    "image_id": 1,
    "category_id": 1,
    "price_bbox": [
        402,
        652,
        37,
        19
    ],
    "match_points": [
        [
            421,
            650
        ],
        [
            420,
            652
        ]
    ],
    "match_length": "48.0px",
    "promo": "promo",
    "sku_price": "100"
},
...
]
```

If sku and price_tag are not matched, sku annotation doesn't get updated:

```
"annotations": 
[
...,
{
    "id": 10,
    "bbox": [
        351,
        577,
        38,
        73
    ],
    "image_id": 1,
    "category_id": 9
},
...
]
```