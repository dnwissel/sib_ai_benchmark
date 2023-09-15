from hiclass.metrics import f1

def f1_hier(y_true, y_pred, en):
    y_true_encoded = en.transform(y_true)
    # print(y_true_encoded, y_pred)
    score = f1(y_true_encoded, y_pred)
    return score