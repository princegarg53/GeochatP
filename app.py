import torch
import gradio as gr
from torchvision import transforms
from PIL import Image

# Load model
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers here

    def forward(self, x):
        # Forward pass
        return x

model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define prediction function
def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    return output.numpy().tolist()

# Create Gradio interface
iface = gr.Interface(fn=predict, inputs=gr.Image(), outputs="json")
iface.launch()
