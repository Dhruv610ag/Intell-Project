import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.vgg import vgg16
import os
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

class BSDS300Dataset(Dataset):
    def __init__(self, image_dir, list_file, transform=None):
        self.image_dir = image_dir
        with open(list_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = default_loader(img_path)  # PIL image
        if self.transform:
            image = self.transform(image)
        return image


# 1. Simple CNN-based Teacher and Student
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.encoder(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.encoder(x)

# 2. Loss Functions
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, use_perceptual_loss=False):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.use_perceptual_loss = use_perceptual_loss
        
        if self.use_perceptual_loss:
            # Load pre-trained VGG16 model for perceptual loss
            self.vgg16 = vgg16(pretrained=True).features[:16].eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            for param in self.vgg16.parameters():
                param.requires_grad = False

    def forward(self, student_output, teacher_output, target):
        loss_l1 = self.l1(student_output, target)
        loss_l2 = self.l2(student_output, target)
        distill_loss = self.l2(student_output, teacher_output)
        
        if self.use_perceptual_loss:
            # Perceptual loss (VGG16)
            vgg_target = self.vgg16(target)
            vgg_student_output = self.vgg16(student_output)
            perceptual_loss = self.l2(vgg_student_output, vgg_target)
        else:
            perceptual_loss = 0
        
        return self.alpha * distill_loss + 0.2 * loss_l1 + 0.1 * loss_l2 + 0.1 * perceptual_loss

# 3. Dataset (use BSDS500 or DIV2K or your own image sharpening pairs)
def get_dataloaders(batch_size=8, image_size=128):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    train_dataset = BSDS300Dataset(
        image_dir='BSDS300/images',
        list_file='BSDS300/iids_train.txt',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


# 4. Training
def train_epoch(teacher, student, loader, optimizer, criterion, device):
    student.train()
    teacher.eval()
    total_loss = 0
    for batch in loader:
        imgs, _ = batch
        imgs = imgs.to(device)
        sharp_imgs = imgs  # Assuming input=output for FakeData

        with torch.no_grad():
            teacher_preds = teacher(imgs)
        student_preds = student(imgs)
        loss = criterion(student_preds, teacher_preds, sharp_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# 5. Main Function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = TeacherModel().to(device)
    student = StudentModel().to(device)

    # Optional: load pretrained teacher
    # teacher.load_state_dict(torch.load("teacher.pth"))

    dataloader = get_dataloaders()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = DistillationLoss(use_perceptual_loss=True)  # Set to True if you want to include perceptual loss

    # Pretrain teacher if needed
    teacher.eval()  # Already trained

    for epoch in range(5):
        loss = train_epoch(teacher, student, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        torch.save(student.state_dict(), f"student_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
