from sklearn.metrics import r2_score

def adjusted_r2_score(y_true, y_pred, n, p):
    """
    n - number of observations
    p - number of independent variables
    """
    return 1 - (1 - r2_score(y_true, y_pred))*((n-1)/(n - p - 1))
