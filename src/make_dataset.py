import torch
import glob
from PIL import Image
from torchvision import transforms, models
import os
from matplotlib import pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm, tqdm_notebook
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import argparse

plt.rcParams["savefig.bbox"] = 'tight'

def segment_image(folder_path):
    model = load_model()
    #furniture_types = ["chair","bed","sofa","swivelchair","table"]
    furniture_types = ["chair"]

    for furniture_type in furniture_types:
        with torch.no_grad():
            for file_path in glob.glob(folder_path + furniture_type + "/" + '*.jpg'):
                input_batch, input_image = preprocess(file_path)
                output = model(input_batch)['out'][0]            
                output_predictions = np.array(output.argmax(0))
                
                idx = (output_predictions!=9)
                input_image = np.array(input_image)
                input_image[idx] = 0
                plt.imsave(os.getcwd() + "/data/seg_chairs/" + file_path.split("/")[-1],input_image)

        
    plt.figure(figsize=(15,15))
    plt.imshow(r)
    plt.figure(figsize=(15,15))
    plt.imshow(input_image)
    plt.show()


def load_model():
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    if torch.cuda.is_available():    
        model.to('cuda')
    return model

def preprocess(file_path):
    input_image = Image.open(file_path)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    return input_batch, input_image

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
        
def load_resnet():
    model = models.resnet152(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    if torch.cuda.is_available():  
        print("CUDA AVAILABLE")  
        model.to('cuda')
    return model

def extract_features(folder_path):
    model = load_resnet()
    furniture_types = ["chair","bed","sofa","swivelchair","table"]
    #furniture_types = ["chair"]


    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    features_dict = dict()
    for furniture_type in tqdm(furniture_types):
        with torch.no_grad():
            for file_path in tqdm(glob.glob(folder_path + furniture_type + "/" + '*.jpg')):
                    input_image = Image.open(file_path)
                    input_image = input_image.convert("RGB")
                    input_tensor = preprocess(input_image)
                    input_batch = input_tensor.unsqueeze(0)
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                    features = model(input_batch)
                    features = features.flatten()
                    normalized_features = features / norm(features)
                    features_dict[file_path.split("/")[-2] + "/" + file_path.split("/")[-1]] = list(normalized_features.cpu().numpy())

                    
        
    pickle.dump(features_dict, open('data/features/features_03042022.pickle', 'wb'))
    
def fit_save_nn(dict_path='data/features/features_03042022.pickle',save_path="models/nn.pk"):
    features_dict = pickle.load(open(dict_path, 'rb'))
    feature_list = list(features_dict.values())
    feature_keys = np.array(list(features_dict.keys()))
    n_n = NearestNeighbors(n_neighbors=3, algorithm='auto',
    metric='euclidean').fit(feature_list)
    with open(save_path,"wb") as filehandler:
        pickle.dump(n_n, filehandler)
    print("Saved NN to: " + str(save_path))
    
def load_nn(model_path="models/nn.pk"):
    return pickle.load(open(model_path, 'rb'))

def find_nn(feature,model,dict_path='data/features/features_03042022.pickle',plot=False):
    features_dict = pickle.load(open(dict_path, 'rb'))
    feature_list = list(features_dict.values())
    feature_keys = np.array(list(features_dict.keys()))
    
    distances, indices = model.kneighbors(feature)
    inds = [0,*indices[0]]
    dists = [0,*distances[0]]
    if plot:
        plot_images(feature_keys[inds],distances[0])
    return inds,dists

def load_pca(model_path="models/pca.pk"):
    return pickle.load(open(model_path, 'rb'))

def pca_transform(feature,model):
    feature_compressed = model.transform(np.array(feature).reshape(1,-1))
    return list(feature_compressed[0])   
    
def fit_save_pca(dict_path='data/features/features_03042022.pickle',save_path="models/pca.pk"):
    features_dict = pickle.load(open(dict_path, 'rb'))
    feature_list = list(features_dict.values())
    num_feature_dimensions=100
    pca = PCA(n_components = num_feature_dimensions)
    pca.fit(feature_list)
    with open(save_path,"wb") as filehandler:
        pickle.dump(pca, filehandler)
    print("Saved PCA to: " + str(save_path))
    
def save_transformed(features_dict,save_path='data/features/features_compr_03042022.pickle'):
    pickle.dump(features_dict, open(save_path, 'wb'))

def plot_images(filenames, distances):
    images = []
    for file_path in filenames:
        images.append(Image.open(os.getcwd() + "/data/train/" + file_path))
    plt.figure(figsize=(20, 14))
    columns = 4
    for i, image in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + filenames[i].split("/")[-1])
        else:
            ax.set_title("Similar Image\n" + filenames[i].split("/")[-1] +
                         "\nDistance: " +
                         str(float("{0:.2f}".format(distances[i-1]))))
        plt.imshow(image)
    plt.show()
        
if __name__ == "__main__":
    cwd_path = os.getcwd()
    parser = argparse.ArgumentParser(description="Make dataset and Nearest Neighbor functions")
    parser.add_argument("--dict_path", default=cwd_path + '/data/features/features_03042022.pickle',help="dictionary path")
    parser.add_argument("--nn_model_path", default=cwd_path + "/models/pca.pk",help="Nearest Neighbor model path")
    parser.add_argument("--pca_model_path", default=cwd_path + "/models/pca.pk",help="PCA model path")
    parser.add_argument("--data_folder_path", default=cwd_path + "/data/train/",help="Data folder path")
    parser.add_argument("--dict_path", default=cwd_path + 'data/features/features_03042022.pickle',help="dictionary path")
    
    parser.add_argument("--nn", default=False,help="Run Nearest Neighbors Search")
    parser.add_argument("--feature_path", default=cwd_path + 'data/input/img1.jpg',help="input feature (jpg) path")

    args = parser.parse_args()
    
    train_path = os.path.join(os.getcwd() + "/data/train/")
    #extract_features(train_path)
    #find_nn('data/features/features_compr_03042022.pickle')
    fit_save_nn()