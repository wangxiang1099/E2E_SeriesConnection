
# 这个模型裁剪 相当于 在一个大的网络里采样小的子网络 目前不做实现

def modelClipper(model):

    del model.cnn
    
    model.forward_network = lambda x: model.remain(x)
    # test_model  
    return model 