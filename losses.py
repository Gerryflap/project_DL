

def mse_loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).sum(dim=0) / len(y_pred)

