from PIL import Image
import numpy as np
import json
import logging


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def convert(input_file):
    try:
        img = Image.open(input_file)
        img = img.convert('RGB')
        img = img.resize((256, 256), Image.NEAREST)
        image = np.asarray(img)
        image = np.true_divide(image, [255.0], out=None)
        image = np.reshape(image, (1, 256, 256, 3))
        image = image.astype(np.float32)
        data = json.dumps(image, cls=NumpyEncoder)
        data = data[1:-1]
    except Exception as err:
        msg = "Failed to convert input image. " + str(err)
        logging.error(msg)
        return "", msg
    return eval(data), ""
