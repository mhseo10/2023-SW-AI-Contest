import os
import time
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from helper import *
from timm.scheduler.cosine_lr import CosineLRScheduler
from segmentation_models_pytorch import DeepLabV3Plus

train_transform = A.Compose([
    A.RandomCrop(224, 224),
    A.Normalize(),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

epochs = 100

if __name__ == "__main__":
    set_seed(7777)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists('./train.parquet'):
        make_parquet('./train.csv')

    dataset = SatelliteDataset(path='./train.parquet', transform=train_transform, shape=(1024, 1024))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=0)

    # model 초기화
    model = DeepLabV3Plus(encoder_name='timm-efficientnet-b3', encoder_weights='noisy-student', classes=1)
    model.to(device)

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=epochs,
                                  lr_min=0.00001,
                                  cycle_mul=1,
                                  cycle_decay=0.5,
                                  cycle_limit=1,
                                  warmup_t=5,
                                  warmup_lr_init=0.00001)

    train_loss = []

    # training loop
    for epoch in range(100):
        with tqdm(dataloader, unit='batch') as t_loader:
            t_loader.set_description(f'Epoch {epoch + 1}')

            epoch_loss = 0
            total_dice_score = []
            model.train()

            for images, masks in dataloader:
                t_loader.update(1)
                images = images.float().to(device)
                masks = masks.float().to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(images)

                    bce_loss = criterion(outputs, masks.unsqueeze(1))
                    dice_loss, dice_score = DiceLoss(outputs, masks.unsqueeze(1))

                loss = bce_loss + dice_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                total_dice_score.append(dice_score)

            train_loss.append(epoch_loss / len(dataloader))
            total_dice_score = torch.mean(torch.concat(total_dice_score))
            t_loader.set_postfix({'loss': train_loss[-1], 'dice_score': total_dice_score.detach().item()})
            t_loader.close()

    # test loop
    test_dataset = SatelliteDataset(path='./test.csv', transform=test_transform, shape=(224, 224), infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=0)

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)

            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8)  # Threshold = 0.35

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '':  # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)

    submit = pd.read_csv(f'./sample_submission.csv')
    submit['mask_rle'] = result

    submit.to_csv(f'./{time.strftime("%m%d_%H%M", time.localtime(time.time()))}_submit.csv', index=False)
