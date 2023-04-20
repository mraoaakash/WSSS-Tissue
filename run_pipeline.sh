##  Shell code for running the whole pipeline.
## Modifying these two variables to replace the experimental dataset.
## DATASET is the abbreviation for dataset in our project and DATASETNAME is the folder name of the dataset.
DATASET=luad
DATASETNAME=LUAD-HistoSeg
# DATASET=bcss
# DATASETNAME=BCSS-WSSS

##  Run Stage1: To train and test the ResNet38-based multi-label classification model.
python 1_train_stage1.py                    \
    # --dataset luad                      \
    # --trainroot datasets/LUAD-HistoSeg/train/\
    # --testroot datasets/LUAD-HistoSeg/test/  \
    # --max_epoches 20                        \
    >> logs/stage1_log_on_$DATASET.txt
##  Generate Pseudo-Mask: Generate 3-level PMs by above model of Stage1.
python 2_generate_PM.py                       \
    --dataroot datasets/LUAD-HistoSeg        \
    --dataset luad                      \
    --weights checkpoints/stage1_checkpoint_trained_on_$DATASET.pth\
    >> logs/stagePM_log_on_$DATASET.txt
#  Run Stage2: Train the deeplab v3+ model with 3-level pseudo-mask.
python 3_train_stage2.py                    \
    --dataset luad                      \
    --dataroot datasets/LUAD-HistoSeg        \
    --epochs 30                              \
    --Is_GM False                           \
    --resume_stage1 checkpoints/stage1_checkpoint_trained_on_$DATASET.pth\
    --resume init_weights/deeplab-resnet.pth.tar\
    >> logs/stage2_log_on_$DATASET.txt