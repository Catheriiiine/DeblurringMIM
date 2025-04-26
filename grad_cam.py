import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

# Load your model (example: convvit_base_patch16)
from model.models_convvit import convvit_base_patch16  # adjust import as needed

# Define a wrapper for patch_embed4 to ensure a 4D output.
class PatchEmbedWrapper(torch.nn.Module):
    def __init__(self, patch_embed):
        super().__init__()
        self.patch_embed = patch_embed
        
    def forward(self, x):
        # Get the output from the original patch embedding layer.
        x_out = self.patch_embed(x)
        # If the output is 3D, reshape it.
        if x_out.dim() == 3:
            # Assume x_out has shape [B, N, C] where N = H*W.
            B, N, C = x_out.shape
            H = W = int(N ** 0.5)
            # Reshape to [B, C, H, W]
            x_out = x_out.transpose(1, 2).view(B, C, H, W)
        return x_out

def reshape_transform(x):
    # x is expected to have shape [B, N, C]
    B, N, C = x.shape
    H = W = int(N ** 0.5)
    return x.transpose(1, 2).view(B, C, H, W)

model = convvit_base_patch16(num_classes=2, drop_path_rate=0.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the pre-trained checkpoint
checkpoint_path = "/home/catherine/Desktop/DeblurringMIM/cls_ckpt/checkpoint-40.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

state_dict = checkpoint['model']
new_state_dict = {}
for k, v in state_dict.items():
    # Replace "fc_norm" with "norm" if that's the only difference:
    new_key = k.replace("fc_norm.", "norm.")
    new_state_dict[new_key] = v
# Adjust the key below depending on how your checkpoint is structured
model.load_state_dict(new_state_dict)

model.eval()

for name, module in model.named_children():
    print(name)

# Choose the target layer for Grad-CAM.
# Adjust the target_layer based on your model structure.
print(model.patch_embed4)
print(model.blocks3)

target_layer = model.patch_embed4  # example target layer

# Initialize Grad-CAM
cam = GradCAM(
    model=model,
    target_layers=[model.patch_embed4],
    use_cuda=torch.cuda.is_available(),
    reshape_transform=reshape_transform
)

# Alternatively, you can use GradCAMPlusPlus:
# cam = GradCAMPlusPlus(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

# Define preprocessing: update normalization values if needed
preprocess = transforms.Compose([
    transforms.Resize((224,224)),  # Add this line to resize input image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load an ultrasound image you want to visualize
img_path = "/home/catherine/Desktop/DeblurringMIM/blurred_fold_3/test/negative/001-02.09.2020 10.24.32-02.09.2020 10.24.32-6_1.jpg"
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise ValueError("Image not found at: " + img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb_resized = cv2.resize(img_rgb, (224, 224))
cv2.imwrite("/home/catherine/Desktop/DeblurringMIM/neg_image_2.jpg", cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2BGR))

img_float = img_rgb_resized.astype(np.float32) / 255.0# for visualization overlay

pil_img = Image.fromarray(img_rgb)

input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

target_class = 1  # or any valid class index
targets = [ClassifierOutputTarget(target_class)]
targets = None

# Run Grad-CAM
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]  # Get CAM for the first image

# Overlay CAM on the image
visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
plt.title("Grad-CAM Visualization")
plt.axis('off')
plt.show()

# Optionally, save the visualization:
cv2.imwrite("/home/catherine/Desktop/DeblurringMIM/neg_salience_2.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
