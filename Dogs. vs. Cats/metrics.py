# STAN
def log_loss(y_true, y_pred):
    epsilon = 1e-15
    #y_pred = K.clip(y_pred, epsilon, 1-epsilon)
    margin = 0.025
    y_pred = K.clip(y_pred, margin, 1-margin)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out)

