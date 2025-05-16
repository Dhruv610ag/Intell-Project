import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.utils import save_image
from torchvision.datasets.folder import default_loader
import os

# Dataset Class
class BSDS300Dataset(Dataset):
    def __init__(self, image_dir, list_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []

        with open(list_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]

        extensions = ['jpg', 'jpeg', 'png']
        for image_id in image_ids:
            for ext in extensions:
                img_path = os.path.join(self.image_dir, f"{image_id}.{ext}")
                if os.path.exists(img_path):
                    self.image_paths.append((image_id, img_path))
                    break

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_id, image_path = self.image_paths[idx]
        image = default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image_id, image

# Student Model (ResNet50)
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()  # Remove final classification layer

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)
        features = self.model.layer1(features)
        features = self.model.layer2(features)
        features = self.model.layer3(features)
        features = self.model.layer4(features)
        return self.decoder(features)

# Teacher Model (ResNet50x2)
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(base_model.children())[:-2])  # Remove avgpool and fc
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.model(x)
        return self.decoder(features)

# Loss Function with Perceptual Loss
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, use_perceptual_loss=False):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.use_perceptual_loss = use_perceptual_loss

        if self.use_perceptual_loss:
            self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
            for param in self.vgg16.parameters():
                param.requires_grad = False
            self.vgg16.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, student_output, teacher_output, target):
        loss_l1 = self.l1(student_output, target)
        loss_l2 = self.l2(student_output, target)
        distill_loss = self.l2(student_output, teacher_output)
        perceptual_loss = 0

        if self.use_perceptual_loss:
            vgg_target = self.vgg16(target)
            vgg_student = self.vgg16(student_output)
            perceptual_loss = self.l2(vgg_student, vgg_target)

        return self.alpha * distill_loss + 0.2 * loss_l1 + 0.1 * loss_l2 + 0.1 * perceptual_loss

# Data Loader
def get_dataloader(batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((1080, 1920)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor()
    ])

    dataset = BSDS300Dataset(
        image_dir=r'C:\Users\Ayush Sharma\OneDrive\Desktop\intel\Intell-Project\BSDS300\images\train',
        list_file=r'C:\Users\Ayush Sharma\OneDrive\Desktop\intel\Intell-Project\BSDS300\iids_train.txt',
        transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Save outputs
def save_outputs(student_model, dataloader, device, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    student_model.eval()

    with torch.no_grad():
        for image_id_batch, imgs in dataloader:
            imgs = imgs.to(device)
            outputs = student_model(imgs)
            outputs = F.interpolate(outputs, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            for image_id, output in zip(image_id_batch, outputs):
                save_image(output.cpu(), os.path.join(output_dir, f"{image_id}.png"))

# Training Function
def train(teacher, student, dataloader, optimizer, criterion, device):
    student.train()
    teacher.eval()
    total_loss = 0

    for _, imgs in dataloader:
        imgs = imgs.to(device)
        with torch.no_grad():
            teacher_preds = teacher(imgs)

        student_preds = student(imgs)

        # Resize outputs to match input resolution
        teacher_preds = F.interpolate(teacher_preds, size=imgs.shape[2:], mode='bilinear', align_corners=False)
        student_preds = F.interpolate(student_preds, size=imgs.shape[2:], mode='bilinear', align_corners=False)

        loss = criterion(student_preds, teacher_preds, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = TeacherModel().to(device)
    student = StudentModel().to(device)

    dataloader = get_dataloader(batch_size=1)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = DistillationLoss(use_perceptual_loss=True)

    for epoch in range(10):
        loss = train(teacher, student, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")

    save_outputs(student, dataloader, device)

if __name__ == "__main__":
    main()
