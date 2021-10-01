from sklearn.metrics import *


def rnd(x, digits=5):
    '''Helper to make rounding consistent '''
    return round(x, ndigits=digits)


def output_metrics(y_true, y_pred, dataset_label, digits=5):
    '''Output the usual performance metrics and a classification report'''
    msg = f'''
    --- {dataset_label} ---
    {classification_report(y_true, y_pred, digits=digits)}

    Accuracy:   {rnd(accuracy_score(y_true, y_pred), digits=digits)}
    Precision:  {rnd(precision_score(y_true, y_pred, average='macro'), digits=digits)}
    Recall:     {rnd(recall_score(y_true, y_pred, average='macro'), digits=digits)}
    F-score:    {rnd(f1_score(y_true, y_pred, average='macro'), digits=digits)}
    '''

    return msg
