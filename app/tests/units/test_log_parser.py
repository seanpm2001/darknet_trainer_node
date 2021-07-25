from log_parser import LogParser
from uuid import uuid4

data = """

 (next mAP calculation at 1000 iterations)
 1: 46.133102, 58.463966 avg loss, 3e-06 rate, 70.404934 seconds, 135808 images, 11070.078691 hours left
 total_bbox = 142207, rewritten_bbox = 0.026018 %
Loaded: 0.000024 seconds

 (next mAP calculation at 1000 iterations)
 2: 109.290443, 99.471283 avg loss, 0.000001 rate, 70.404934 seconds, 135808 images, 11070.078691 hours left

1Total Detection Time: 0 Seconds
Saving weights to backup//tiny_yolo_best_mAP_0.000000_iteration_1000_avgloss_-nan_.weights
Saving weights to backup//tiny_yolo_1000_avgloss_-nan_.weights
Saving weights to backup//tiny_yolo_last.weights

Resizing to initial size: 32 x 32  try to allocate additional workspace_size = 0.15 MB
 CUDA allocate done!

calculation mAP (mean average precision)...
 Detection layer: 16 - type = 28
 Detection layer: 23 - type = 28

 detections_count = 0, unique_truth_count = 3
class_id = 0, name = purple, ap = 37.00%          (TP = 34, FP = 7, FN = 8)
class_id = 1, name = green, ap = 0.00%           (TP = 12, FP = 6, FN = 14)

 for conf_thresh = 0.25, precision = -nan, recall = 0.00, F1-score = -nan
 for conf_thresh = 0.25, TP = 0, FP = 0, FN = 3, average IoU = 0.00 %

 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall
 mean average precision (mAP@0.50) = 0.793866, or 0.00 %

Set -points flag:
 `-points 101` for MS COCO
 `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data)
 `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset

 mean_average_precision (mAP@0.5) = 0.000000
Loaded: 0.061675 seconds
 (next mAP calculation at 1001 iterations)

 (next mAP calculation at 46308 iterations)
 Last accuracy mAP@0.5 = 94.33 %, best = 94.82 %
 46217: 0.107458, 0.144404 avg loss, 0.000100 rate, 1.164577 seconds, 2957888 images, 12.510687 hours left
 total_bbox = 8079425, rewritten_bbox = 2.030293 %
 total_bbox = 8079433, rewritten_bbox = 2.030303 %
 total_bbox = 8079453, rewritten_bbox = 2.030311 %
 total_bbox = 8079486, rewritten_bbox = 2.030315 %
 total_bbox = 8079498, rewritten_bbox = 2.030312 %
 total_bbox = 8079524, rewritten_bbox = 2.030330 %
 total_bbox = 8079545, rewritten_bbox = 2.030325 %
 total_bbox = 8079573, rewritten_bbox = 2.030342 %

 Tensor Cores are used.
Loaded: 0.000046 seconds

 (next mAP calculation at 46308 iterations)
 Last accuracy mAP@0.5 = 94.33 %, best = 94.82 %
 46218: 0.202720, 0.150236 avg loss, 0.000100 rate, 1.228593 seconds, 2957952 images, 12.494870 hours left
 total_bbox = 8079603, rewritten_bbox = 2.030360 %
 total_bbox = 8079653, rewritten_bbox = 2.030360 %
 total_bbox = 8079673, rewritten_bbox = 2.030354 %
 total_bbox = 8079685, rewritten_bbox = 2.030351 %
 total_bbox = 8079689, rewritten_bbox = 2.030350 %
 total_bbox = 8079714, rewritten_bbox = 2.030344 %
 total_bbox = 8079729, rewritten_bbox = 2.030340 %
 total_bbox = 8079744, rewritten_bbox = 2.030337 %

 """

parser = LogParser(LogParser.extract_iteration_log(data))


def test_extract_iteration_log():
    iteration_log = LogParser.extract_iteration_log(data)
    assert iteration_log[1].startswith(
        ' 2: 109.290443, 99.471283 avg loss, 0.000001 rate, 70.404934 seconds, 135808 images, 11070.078691 hours left')
    assert iteration_log[-1] == 'Loaded: 0.061675 seconds'


def test_parsing_empty_log():
    iteration_log = LogParser.extract_iteration_log('')
    assert iteration_log == []
    p = LogParser(iteration_log)
    mAP = p.parse_mAP()
    assert mAP == None
    parsed_class = p.parse_classes()
    assert parsed_class == []

def test_parse_mAP():
    mAP = parser.parse_mAP()
    expected_mAP = {'mAP': 0.793866, 'mAP_percentage': 0.5}

    assert mAP == expected_mAP


def test_parse_iteration():
    iteration = parser.parse_iteration()
    assert iteration == 2


def test_parse_classes():
    parsed_class = parser.parse_classes()
    assert len(parsed_class) == 2
    assert parsed_class[0]['name'] == "purple"
    assert parsed_class[1]['name'] == "green"

    purple_class = parsed_class[0]
    assert purple_class['id'] == "0"
    assert purple_class['ap'] == 37
    assert purple_class['tp'] == 34
    assert purple_class['fp'] == 7
    assert purple_class['fn'] == 8


def test_parse_weightfile():
    weight_file = parser.parse_weightfile()
    assert weight_file == 'backup/tiny_yolo_best_mAP_0.000000_iteration_1000_avgloss_-nan_.weights'
