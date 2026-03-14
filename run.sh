#模型训练（水域分割）
# --BACKBONE_TYPE swin_t
# --BACKBONE_TYPE resnet50
python train_water.py `
    --TRAIN_IMAGE_DIR "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\train\images" `
    --TRAIN_MASK_DIR "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\train\masks" `
    --VAL_IMAGE_DIR "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\val\images" `
    --VAL_MASK_DIR "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\val\masks" `
    --MODEL_TYPE upernet `
    --BACKBONE_TYPE swin_t `
    --BANDS 3 `
    --NUM_CLASS 2 `
    --IMG_SIZE 256 `
    --BATCH_SIZE 16 `
    --EPOCHS 10 `
    --OPTIMIZER_TYPE sgd `
    --LOSS_TYPE ce `
    --LR_SCHEDULER poly `
    --INIT_LR 0.0005 `
    --GPU_ID 0


#模型测试
python test_water.py `
    --TEST_IMAGE_DIR "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\test\images" `
    --TEST_MASK_DIR "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\test\masks" `
    --MODEL_TYPE upernet `
    --BACKBONE_TYPE swin_t `
    --BANDS 3 `
    --NUM_CLASS 2 `
    --IMG_SIZE 256 `
    --BATCH_SIZE 16 `
    --MODEL_PATH "pth_files/upernet-epoch14-loss0.113-val_loss0.091.pth" `
    --SAVE_PRED True


#模型推理（单张图片）
python predict_water_img.py `
    --IMAGE_PATH "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\test\images\H05_1_0000000950.jpg" `
    --MODEL_TYPE upernet `
    --BACKBONE_TYPE swin_t `
    --BANDS 3 `
    --NUM_CLASS 2 `
    --IMG_SIZE 256 `
    --MODEL_PATH "pth_files/upernet-epoch14-loss0.113-val_loss0.091.pth" `
    --SAVE_PATH "result.png" `
    --SHOW True


#模型推理（文件夹）
python predict_water_folder.py `
    --INPUT_DIR "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\test\images" `
    --OUTPUT_DIR "predictions" `
    --MODEL_TYPE upernet `
    --BACKBONE_TYPE swin_t `
    --BANDS 3 `
    --NUM_CLASS 2 `
    --IMG_SIZE 256 `
    --MODEL_PATH "pth_files/upernet-epoch14-loss0.113-val_loss0.091.pth"