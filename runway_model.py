import torch
import getopt
import math
import numpy as np
import os
import PIL
import PIL.Image
import sys
import tempfile
import flowiz
from correlation import correlation # the custom cost volume layer
import runway
from runway.data_types import *
from run import *
from run import estimate
from pathlib import Path

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True

@runway.setup(options={"model_file": file(extension=".pytorch")})
def setup(opts):
    checkpoint = opts["model_file"]
    network = Network(checkpoint).cuda().eval()
    return network


command_inputs = {"input_image": image}
command_outputs = {"output_image": image}

last_frame = None

@runway.command("compute_flow", inputs=command_inputs, outputs=command_outputs, description="Computes Optical Flow")
def compute_flow(network, inputs):
    global last_frame
    current_frame = np.array(inputs["input_image"])

    if last_frame is None:
        output = np.full(current_frame.shape, 255)
    else:
        tensorFirst = torch.FloatTensor(np.array(last_frame)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
        tensorSecond = torch.FloatTensor(np.array(current_frame)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
        tensorOutput = estimate(network, tensorFirst, tensorSecond)
        output = tensorOutput.numpy().transpose(1, 2, 0)
        output = flowiz.convert_from_flow(output)

    last_frame = current_frame

    return {"output_image": output}

if __name__ == "__main__":
    runway.run()

