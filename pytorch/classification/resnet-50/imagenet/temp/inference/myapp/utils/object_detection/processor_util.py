from PIL import Image
import base64
import os
import json
import numpy 

from myapp.utils.object_detection import visualization_utils as vis_util
from myapp.utils.object_detection import label_map_util

def b64_filewriter(filename, content):
    string = content.encode('utf8')
    b64_decode = base64.b64decode(string)
    fp = open(filename, "wb")
    fp.write(b64_decode)
    fp.close()

def process_output(image_data, label_map_string, result, num_classes):
    b64input = json.loads(image_data)
    data = b64input['signatures']['inputs'][0][0]['data']
    input_file = './temp'
    b64_filewriter(input_file, data)
    img = Image.open(input_file)
    image_np = numpy.array(img)
    os.remove(input_file)

    # Plot boxes on the input image
    label_map = label_map_util.load_labelmap(label_map_string)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    result = json.loads(result)
    boxes = result['predictions'][0]['detection_boxes']
    classes = result['predictions'][0]['detection_classes']
    scores = result['predictions'][0]['detection_scores']
    image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        numpy.squeeze(boxes),
        numpy.squeeze(classes).astype(numpy.int32),
        numpy.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    im = Image.fromarray(image_vis)
    out_file = 'tmp.jpeg'
    im.save(out_file)
    image_data = ""
    with open(out_file, "rb") as imageFile:
       image_data = base64.b64encode(imageFile.read())
    os.remove(out_file)
    return image_data
