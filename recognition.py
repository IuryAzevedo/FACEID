import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models

# Atualize o caminho do modelo treinado
model_path = './face_recognition_model.pth'  # Certifique-se de que o arquivo está no mesmo diretório que este script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_model(model_path, device):
    # Carregar o estado do modelo para obter o número de classes
    state_dict = torch.load(model_path, map_location=device)
    num_classes = state_dict['fc.weight'].shape[0]
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, num_classes

model, num_classes = initialize_model(model_path, device)

# Transformações nos dados
data_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),zip
])


class_names = ['Yuri Nekan', 'Vitoria Medeiros', 'Emanuel Vidal', 
               'Davi Ximenes', 'Rodrigo Lages', 'Alejandro Elias', 
               'Tamires Sousa', 'Pedro Henrique']
class_names = [f'pessoa_{i}' for i in range(num_classes)]  

camera_index = 0
celular = cv2.VideoCapture(camera_index)

# Reconhecimento de Rosto
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

if not celular.isOpened(): 
    print('Não foi possível acessar a câmera do celular!')
    exit()

while True:
    verificador, frame = celular.read()
    if not verificador:
        print("Erro ao capturar o frame.")
        break

    lista_rostos = reconhecedor_rostos.process(frame)
    
    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            bboxC = rosto.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Recorta a região do rosto
            face_img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            if face_img.size == 0:
                continue

            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img_pil = Image.fromarray(face_img)
            face_img_tensor = data_transforms(face_img_pil)
            face_img_tensor = face_img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(face_img_tensor)
                _, preds = torch.max(outputs, 1)
                label = class_names[preds[0].item()]

            # Desenha a caixa delimitadora e o nome
            cv2.rectangle(frame, bbox, (255, 0, 0), 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            desenho.draw_detection(frame, rosto)

    cv2.imshow('Camera do Celular', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

celular.release()
cv2.destroyAllWindows()
