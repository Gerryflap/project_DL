

def mse_loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).mean(dim=0)

