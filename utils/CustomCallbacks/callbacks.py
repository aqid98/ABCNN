def CustomLearningRateScheduler(epoch, learning_rate):
    '''
    Custom Learning rate scheduler for better training
    References:
    - [Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
    '''
    if epoch<=2:
        return learning_rate
    lr_decay = learning_rate / (epoch)
    return lr_decay