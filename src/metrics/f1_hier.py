from hiclass.metrics import f1, recall, precision

def f1_hier(y_true, y_pred, en):
    y_true_encoded = en.transform(y_true)
    # print(y_true_encoded, y_pred)
    score = f1(y_true_encoded, y_pred)
    return score

def recall_hier(y_true, y_pred, en):
    y_true_encoded = en.transform(y_true)
    # print(y_true_encoded, y_pred)
    score = recall(y_true_encoded, y_pred)
    return score

def precision_hier(y_true, y_pred, en):
    y_true_encoded = en.transform(y_true)
    # print(y_true_encoded, y_pred)
    score = precision(y_true_encoded, y_pred)
    return score