## Reference: https://github.com/facebookresearch/suncet/blob/master/configs/paws/cifar10_train.yaml

############# Path to data #######
SOURCE_DS_TRAIN = "../../data/OCT2017/train"
SOURCE_DS_TEST  = "../../data/OCT2017/test"


############ Dataset ############
MULTICROP_BS = 64
SUPPORT_BS = 64
SUPPORT_SAMPLES = 30
SUP_VIEWS = 2
SUPPORT_IDX = "random_idx.npy"

############ Pre-training ############
LABEL_SMOOTHING = 0.1
PRETRAINING_EPOCHS = 2
START_LR = 0.8
WARMUP_LR = 3.2
PRETRAINING_PLOT = "pretraining_ce_loss.png"
PRETRAINED_MODEL = "paws_encoder"

############ Fine-tuning ############
FINETUNING_EPOCHS = 2
FINETUNED_MODEL = "paws_finetuned"
