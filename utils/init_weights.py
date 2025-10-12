# Random (Gaussian) weights initialization
def init_weights(m, mean=0.0, std=0.01):
    """
    Initializes model `m` with Gaussian weights.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)