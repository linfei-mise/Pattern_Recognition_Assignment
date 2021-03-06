import torch
from model import convnext_base
import matplotlib.pyplot as plt
import numpy as np


# create model
model = convnext_base(num_classes=2)
# load model weights
model_weight_path = "weights/best_model.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)

weights_keys = model.state_dict().keys()
for key in weights_keys:
    # remove num_batches_tracked para(in bn)
    if "num_batches_tracked" in key:
        continue
    # [kernel_number, kernel_channel, kernel_height, kernel_width]
    weight_t = model.state_dict()[key].numpy()

    # read a kernel information
    # k = weight_t[0, :, :, :]

    # calculate mean, std, min, max
    weight_mean = weight_t.mean()
    weight_std = weight_t.std(ddof=1)
    weight_min = weight_t.min()
    weight_max = weight_t.max()
    print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                               weight_std,
                                                               weight_max,
                                                               weight_min))

    # plot hist image
    plt.close()
    weight_vec = np.reshape(weight_t, [-1])
    plt.hist(weight_vec, bins=50)
    plt.title(key)
    plt.show()

