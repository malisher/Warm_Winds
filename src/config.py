import string

# Detector constants
det_imgsz = (1280, 1280)
det_conf_thres = 0.6
det_iou_thres = 0.3
det_max_det = 3000

# Recognizer constants
img_w = 160
img_h = 64

alphabet = string.digits + string.ascii_lowercase
num_class = len(alphabet) + 1
num_class_crossed = len(string.digits) + 1

model_lstm_layers = 2
model_lsrm_is_bidirectional = True

regions = ["standard", "promo"]
num_regions = len(regions)

# Drawing constants
font_scale = 0.7
thickness = 2
grey_color = (128, 128, 128)

# Other
padding = 5  # Padding for price tags in Excel
edge_tolerance = 5  # Distance to be considered as edge
