def accuracy_fn(outputs, labels, target=None):
    predicts = nn.functional.softmax(outputs, dim=1).argmax(dim=1)
    predicts = predicts.numpy()
    labels = labels.numpy()
    
    if target is None:
        # Calculate accuracy over all classes.
        return np.sum(predicts == labels) / np.prod(labels.shape)
    else:
        target_mask = (labels == target)
        return np.sum(predicts[target_mask] == target) / np.sum(target_mask)
 
def mean_iou(cnf):
    '''cnf is a confusion matrix where cnf[i,j] represents the number of true labels from class i that have been predicted as class j'''
    ious = []
    for i in range(cnf.shape[0]):
        iou = cnf[i,i] / (np.sum(cnf[i,:]) + np.sum(cnf[:,i]) - cnf[i,i])
        ious.append(iou)
    return np.mean(ious)
