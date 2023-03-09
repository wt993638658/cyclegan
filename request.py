import requests
import torch
from torchvision.utils import save_image
from torchvision import transforms as T
import argparse
import io
from  PIL import Image
PyTorch_REST_API_URL = 'http://127.0.0.1:5000/generator'

def predict_result(image_path):
    image = open(image_path,'rb').read()
    payload = {'image':image}
    r = requests.post(PyTorch_REST_API_URL,files=payload).json()

    if r['success']:
        # save_image(r["generate"], "outputs/B/generate.png")
        a = torch.Tensor(r["generate"])
        save_image(a,"outputs.png")
        # b = bytes(r["generate"],encoding = "utf8")
        # print(type(b))
        # img = Image.open(io.BytesIO(b))
        # img.save("outputs/B/generate.png")
    else :
        print('request failed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cla')
    parser.add_argument('--file',type=str,help='test image file')
    args = parser.parse_args()
    predict_result(args.file)
    # predict_result("C:/Users/232/Desktop/pythonProject/apple2orange/testA/n07740461_41.jpg")