# CVAT excel pipeline

## Installation

```
pip install -r requirements.txt
```

## Example of usage

```
python main.py --data_folder=data/792_annotations/ --annotations_path=data/792_annotations/annotations.xml --brands_path=data/792_annotations/filtered.xlsx --process_by_brands=True
```

## Possible arguments:

--data_folder - Folder containing all data

--annotations_path - Path to annotation

--brands_path (optional) - Path to filtered brands excel file

--process_by_brands - True if need to process brands, else leave blank (if need to process SKU)
