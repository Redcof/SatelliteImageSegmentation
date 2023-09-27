import pathlib

import cv2
import numpy as np


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0
    
    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]
        
        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width
        
        return [[batchSize, numChannels, height, width]]
    
    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


class LoadNetwork:
    _instance = None
    net = None
    
    def __new__(cls):
        if cls._instance is None:
            path = pathlib.Path(__file__).parents[0]
            cls._instance = super(LoadNetwork, cls).__new__(cls)
            cls.net = cv2.dnn.readNetFromCaffe(str(path / "deploy.prototxt"),
                                               str(path / "hed_pretrained_bsds.caffemodel"))
            cv2.dnn_registerLayer('Crop', CropLayer)
        return cls._instance


def hed(image):
    # Load the model.
    net = LoadNetwork().net
    width, height, _ = image.shape
    inp = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(width, height),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv2.resize(out, (image.shape[1], image.shape[0]))
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)
    return out
