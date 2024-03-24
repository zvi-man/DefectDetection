import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from ml_method.loss import DiceLoss
from ml_method.transform import get_image_and_mask_transforms
from ml_method.unet import load_pretrained_model
from synthetic_data_creation.synthetic_dataset import DefectDataset
from utilities.general_utils import GeneralUtils


def train():
    # load reference image
    reference_img_path = r"../data/defective_examples/case1_reference_image.tif"
    num_images = 4000
    reference_im = GeneralUtils.load_and_display_tiff_image(
        tiff_image_path=reference_img_path,
        to_display=True)
    # Create the dataset
    dataset = DefectDataset(reference_im, num_images=num_images, transform=get_image_and_mask_transforms())

    # Split dataset into train and val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Assuming you have dataloaders for train and val datasets
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_pretrained_model()
    model = model.to(device)
    print(f"Working on Device {device}")
    num_epochs = 20

    criterion = DiceLoss()  # Mean Squared Error Loss

    # Assuming you have optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_train = []
    loss_valid = []

    # Training loop
    for epoch in range(num_epochs):
        loss_train = []
        loss_val = []
        model.train()
        for i, (images, masks) in tqdm(enumerate(train_dataloader)):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())

        # Calculate validation loss
        model.eval()
        with torch.no_grad():
            for i, (images, masks) in tqdm(enumerate(val_dataloader)):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss_val.append(loss.item())

            train_loss_mean = np.mean(loss_train)
            val_loss_mean = np.mean(loss_val)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss_mean}, Val Loss: {val_loss_mean}")


if __name__ == "__main__":
    train()
