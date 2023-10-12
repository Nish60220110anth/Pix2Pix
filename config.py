import lupa
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

lua = lupa.LuaRuntime(unpack_returned_tuples=True)
lua.execute("dofile('config.lua')")

LEARNING_RATE = lua.globals().LEARNING_RATE
BATCH_SIZE = lua.globals().BATCH_SIZE
IMAGE_SIZE = lua.globals().IMAGE_SIZE
NUM_WORKERS = lua.globals().NUM_WORKERS
CHANNELS_IMG = lua.globals().CHANNELS_IMG
L1_LAMBDA = lua.globals().L1_LAMBDA
NUM_EPOCHS = lua.globals().NUM_EPOCHS
LOAD_MODEL = lua.globals().LOAD_MODEL
SAVE_MODEL = lua.globals().SAVE_MODEL
CHECKPOINT_DISC = lua.globals().CHECKPOINT_DISC
CHECKPOINT_GEN = lua.globals().CHECKPOINT_GEN
DEVICE = lua.globals().DEVICE
TRAIN_DIR = lua.globals().TRAIN_DIR
VAL_DIR = lua.globals().VAL_DIR

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},is_check_shapes=False,
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                    0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                    0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
