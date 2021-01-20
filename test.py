if __name__ == '__main__':
     # for training only, need nightly build pytorch
    CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b4_ns',
    'img_size': 512,
    'epochs': 10,
    'train_bs': 32,
    'valid_bs': 32,
    'lr': 1e-4,
    'num_workers': 4,
    'accum_iter': 1, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'tta': 3,
    'used_epochs': [6,7,8,9],
    'weights': [1,1,1,1]
}

    seed_everything(CFG['seed'])
    
    test = pd.DataFrame()
    test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
    test_ds = CassavaDataset(test, '../input/cassava-leaf-disease-classification/test_images/', transforms=get_inference_transforms(), output_label=False)
    

    
    tst_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    device = torch.device(CFG['device'])
    model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique()).to(device)
    
    tst_preds = []
    
    #for epoch in range(CFG['epochs']-3):
    for i, epoch in enumerate(CFG['used_epochs']):    
        model.load_state_dict(torch.load('../input/pytorch-efficientnet-baseline-train-amp-aug/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)))
        
        with torch.no_grad():
            for _ in range(CFG['tta']):
                tst_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, tst_loader, device)]

    tst_preds = np.mean(tst_preds, axis=0) 
    
    
    del model
    torch.cuda.empty_cache()