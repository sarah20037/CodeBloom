import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=True).to(device).eval()

# Resize only for model input
resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# No resizing for original image
original_transform = transforms.ToTensor()
inv_transform = transforms.ToPILImage()

def fgsm_attack(image_tensor, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbation = epsilon * sign_data_grad
    return torch.clamp(image_tensor + perturbation, 0, 1), perturbation

def protect_image(input_path, output_path, epsilon=2/255):
    if not os.path.exists(input_path):
        return

    # Load original image (full size)
    orig_image = Image.open(input_path).convert("RGB")
    orig_tensor = original_transform(orig_image).unsqueeze(0).to(device)

    # Resize image for model input
    resized_tensor = resize_transform(orig_image).unsqueeze(0).to(device)
    resized_tensor.requires_grad = True

    # Forward pass
    output = model(resized_tensor)
    pred = output.max(1)[1]
    loss = torch.nn.CrossEntropyLoss()(output, pred)

    model.zero_grad()
    loss.backward()
    data_grad = resized_tensor.grad.data

    # Get perturbation from resized image
    _, perturbation = fgsm_attack(resized_tensor, epsilon, data_grad)

    # Upsample perturbation to original size
    perturbation_upsampled = torch.nn.functional.interpolate(
        perturbation, size=orig_tensor.shape[2:], mode='bilinear', align_corners=False
    )

    # Apply to original-size image
    protected_tensor = torch.clamp(orig_tensor + perturbation_upsampled, 0, 1)
    protected_image = inv_transform(protected_tensor.squeeze().cpu())

    protected_image.save(output_path)

if __name__ == "__main__":
    base_folder = r"C:\Users\hp\codebloom"
    input_filename = "photo.jpeg"
    output_filename = "photo_protected.jpeg"

    input_path = os.path.join(base_folder, input_filename)
    output_path = os.path.join(base_folder, output_filename)

    protect_image(input_path, output_path)
