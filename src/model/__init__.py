from .models import PixelNeRFNet


def make_model(conf, compute_activation = False, *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get_string("type", "pixelnerf")  # single
    
    if model_type == "pixelnerf":
        net = PixelNeRFNet(conf, compute_activation = compute_activation, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
