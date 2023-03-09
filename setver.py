import io
import json
import flask
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
from  PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50
from torch.autograd import Variable
from models import Discriminator,Generator
import  torchvision.transforms as transforms
import base64
app = flask.Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model():
    global model
    model = Generator().to(device)
    model.load_state_dict(torch.load("models/netG_A2B184439.pth"))
    model.eval()

def prepare_image(image,target_size):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)
    image = T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(image)
    image = image[None]
    image = image.to(device)
    return Variable(image,volatile=True)

@app.route("/generator",methods=["post"])
def generator():
    data = {"success":False}
    if flask.request.method == 'POST':
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        size = 256
        image = prepare_image(image,target_size=(size,size))
        input_A = torch.ones([1, 3, size, size],
                             dtype=torch.float).to(device)
        real_A = torch.tensor(input_A.copy_(image), dtype=torch.float).to(device)
        fake_B = 0.5 * (model(real_A).data+1.0)
        fake_B = fake_B[0].tolist()
        data["generate"] = fake_B
        data["success"] = True
    return flask.jsonify(data)
if __name__ == '__main__':
    load_model()
    app.run()