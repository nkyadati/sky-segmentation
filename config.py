"""Model config in json format"""

CFG = {
    "data": {
        "data_dir": "datasets/city_scapes/",
        "train_image_dir": "train/original/",
        "train_label_dir": "train/mask/",
        "val_image_dir": "val/original/",
        "val_label_dir": "val/mask/",
        "predictions_dir": "./results/results_cityscapes_fcn/",
        "image_ext": ".png",
        "label_ext": ".png",
        "train_batch_size": 12,
        "val_batch_size": 1,
    },
    "model": {
        "model_name": "fcn",  # Options: u2net, unet, fcn
        "epochs": 50,
        "model_dir": "./saved_models/",
        "save_frequency": 1000
    }
}
