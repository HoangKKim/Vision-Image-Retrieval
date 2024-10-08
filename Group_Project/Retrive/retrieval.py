import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from joblib import load
import matplotlib.patches as patches


# Các đường dẫn data
data_dir = '/kaggle/input/fashion-iq-dataset/fashionIQ_dataset'
model_dir = '/kaggle/input/resnet50-weight-fashion/best_model.pth'
image_dir = os.path.join(data_dir, 'images')
json_dir = os.path.join(data_dir, 'image_splits')

# Các hàm cần thiết

# Hàm chuẩn bị dataset
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
# Đọc danh sách tất cả ảnh
def load_all_image_paths():
    image_paths = []
    for split in ['train', 'val', 'test']:
        for category in ['dress', 'shirt', 'toptee']:
            json_path = os.path.join(json_dir, f'split.{category}.{split}.json')
            image_list = read_json(json_path)
            image_paths.extend([os.path.join(image_dir, image_name + '.jpg') for image_name in image_list])
    return image_paths

#  load image (path of image) from file json (follow category)
def load_image_list(category, split):
    json_path = os.path.join(json_dir, f'split.{category}.{split}.json')
    image_list = read_json(json_path)
    return [os.path.join(image_dir, image_name + '.jpg') for image_name in image_list]

# define class FashionIQDataset 
class FashionIQDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Đọc đặc trưng đã lưu từ file
def load_features_and_labels(features_path, labels_path):
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels

# Hàm trích xuất đặc trưng
def extract_feature(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image)
        feature = feature.view(feature.size(0), -1).cpu().numpy()
    return feature

def find_similar_images(query_image_path, all_features, all_labels, all_image_paths, model, transform, device, top_k=10, metric='cosine'):
    # Trích xuất đặc trưng cho ảnh truy vấn
    query_feature = extract_feature(query_image_path, model, transform, device)
    # Tính toán khoảng cách giữa ảnh truy vấn và tất cả ảnh trong tập dữ liệu
    if metric == 'cosine':
        distances = cosine_similarity(query_feature, all_features)
        distances = 1 - distances  # Chuyển đổi từ similarity sang distance
    elif metric == 'euclidean':
        distances = euclidean_distances(query_feature, all_features)
    else:
        raise ValueError("Metric must be either 'cosine' or 'euclidean'")
    # Lấy top_k ảnh gần nhất
    closest_indices = np.argsort(distances[0])[:top_k]
    closest_images = [all_image_paths[idx] for idx in closest_indices]
    closest_labels = [all_labels[idx] for idx in closest_indices]
    closest_distances = distances[0][closest_indices]
    
    return closest_images, closest_labels, closest_distances

# Hàm dùng trong truy vấn
def showImages(queryImage, result_images, top_k, query_label, predict_label):
    # Hiển thị ảnh truy vấn và top-k ảnh phù hợp nhất
    fig, axes = plt.subplots(1, top_k + 1, figsize=(20, 5))

    # Hiển thị ảnh truy vấn
    input_image_path = queryImage
    axes[0].imshow(Image.open(input_image_path))
    axes[0].set_title('Query Image')
    axes[0].axis('off')

    # Hiển thị top-k ảnh phù hợp
    for ax, img_path in zip(axes[1:], result_images):
        img = Image.open(img_path)
        ax.imshow(img)
        if query_label == predict_label:
            ax.add_patch(patches.Rectangle((0, 0), img.width, img.height, edgecolor='blue', facecolor='none', linewidth=2))
        else:
            ax.add_patch(patches.Rectangle((0, 0), img.width, img.height, edgecolor='red', facecolor='none', linewidth=2))
        ax.axis('off')
    plt.show()

def calculate_accuracy(query_label, similar_labels):
    num_correct = np.sum(np.array(similar_labels) == query_label)
    num_total = len(similar_labels)
    accuracy = num_correct / num_total
    return accuracy

def calculate_precision_at_k(query_label, similar_labels, k):
    # Tính precision tại top-k
    relevant_items = np.array(similar_labels)[:k] == query_label
    precision_at_k = np.sum(relevant_items) / k
    return precision_at_k

def calculate_average_precision(query_label, similar_labels):
    relevant_items = np.array(similar_labels) == query_label
    num_relevant_items = np.sum(relevant_items)
    
    if num_relevant_items == 0:
        return 0.0
    
    # Tính AP bằng cách tính Precision tại mỗi k có kết quả đúng
    precision_sum = 0.0
    correct_retrieved = 0
    
    for k in range(1, len(similar_labels) + 1):
        if relevant_items[k - 1]:
            correct_retrieved += 1
            precision_at_k = correct_retrieved / k
            precision_sum += precision_at_k
    
    # Average Precision là trung bình của Precision@k tại các vị trí đúng
    return precision_sum / num_relevant_items

def calculate_map(query_label, similar_labels):    
    return calculate_average_precision(query_label, similar_labels)


# Hàm truy vấn 
## Hàm đánh giá kết quả truy vấn trên toàn bộ tập test với k = {10, 100, 1000, 10000}

### Truy vấn với mô hình ResNet50 (mô hình ban đầu, chưa cải tiến)
def retrieve(query_image, query_label, top_k):
    # Tải mô hình đã huấn luyện
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=False)
    num_classes = 3
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_dir))
    model.to(device)  
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Đường dẫn đến ảnh truy vấn
    query_image_path = os.path.join(image_dir, query_image + '.jpg')

    # Đọc đặc trưng và nhãn đã lưu
    features_path = '/kaggle/input/features/all_features.npy'
    labels_path = '/kaggle/input/features/all_labels.npy'
    all_features, all_labels = load_features_and_labels(features_path, labels_path)

    # Đọc danh sách tất cả ảnh
    all_image_paths = load_all_image_paths()

    # Tìm và hiển thị các ảnh tương tự
    start_time = time.time() # bắt đầu truy vấn
    top_k_image_paths, predicted_labels, distances = find_similar_images(query_image_path, all_features, all_labels, all_image_paths, model, transform, device, top_k=top_k, metric='euclidean')
    end_time = time.time()
                
    # Tính thời gian truy vấn
    query_time = end_time - start_time

    # Tính accuracy
    accuracy = calculate_accuracy(query_label, predicted_labels)
    
    modified_images = [] # các ảnh sau khi đánh dấu sẽ được deploy
        
    for img_path, predicted_label in zip(top_k_image_paths, predicted_labels):
        img = Image.open(img_path)
        padding = 10
        img_with_border = ImageOps.expand(img, border=padding, fill=(255, 255, 255))
        draw = ImageDraw.Draw(img_with_border)
        
        outline_width = 2
        top_left = (outline_width//2, outline_width//2)
        bottom_right = (img_with_border.width - outline_width//2, img_with_border.height - outline_width//2)
    
        if query_label == predicted_label:
            draw.rectangle([top_left, bottom_right], outline='blue', width=2)
        else:
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            
        modified_images.append(img_with_border)
    
    results = {
        'retrieval_results': modified_images,
        'accuracy': accuracy,
        'query_time': query_time,
        
    }
    
    return results


### Truy vấn với mô hình ResNet50+SVM (mô hình sau khi cải tiến)
def improved_retrieval(image_path, image_label, top_k):

    # dress
    train_images_dress = load_image_list('dress', 'train')
    val_images_dress = load_image_list('dress', 'val')
    test_images_dress = load_image_list('dress', 'test')

    # category - shirt
    train_images_shirt = load_image_list('shirt', 'train')
    val_images_shirt = load_image_list('shirt', 'val')
    test_images_shirt = load_image_list('shirt', 'test')

    # category - top&tee
    train_images_toptee = load_image_list('toptee', 'train')
    val_images_toptee = load_image_list('toptee', 'val')
    test_images_toptee = load_image_list('toptee', 'test')

    dress = train_images_dress + val_images_dress + test_images_dress
    shirt = train_images_shirt + val_images_shirt + test_images_shirt
    toptee = train_images_toptee + val_images_toptee + test_images_toptee
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dress_features = np.load('/kaggle/input/each-category-features/dress_features.npy')
    shirt_features = np.load('/kaggle/input/each-category-features/shirt_features.npy')
    toptee_features = np.load('/kaggle/input/each-category-features/toptee_features.npy')
    
    # tải các mô hình
    svm_model = load('/kaggle/input/svm-model/svm_model.joblib')
    
    # sử dụng mô hình resnet50 với trọng số đã huấn luyện
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 3
    model = models.resnet50(weights = None)  
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # load 
    model.load_state_dict(torch.load(model_dir))
    model.to(device)  
    model.eval()
    
    # đọc và mở ảnh
    image = Image.open(os.path.join(image_dir, image_path + '.jpg')).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image)
        feature = feature.view(feature.size(0), -1).cpu().numpy()
            
    input_feature = feature.reshape(-1)
    input_label = image_label          

    # bắt đầu truy vấn
    start_time = time.time()

    # phân loại (dự đoán nhãn cho ảnh truy vấn) 
    label_predict = svm_model.predict([input_feature])
    if(label_predict == 0):
        database_features = dress_features
        database_labels = 0
        database_images = dress
    elif(label_predict == 1):
        database_features = shirt_features
        database_labels = 1
        database_images = shirt
    else: 
        database_features = toptee_features
        database_labels = 2
        database_images = toptee
        
    # tính độ tương đồng
    similarities = cosine_similarity([input_feature], database_features)[0]
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    end_time = time.time()
            
    # Tính thời gian truy vấn
    query_time = end_time - start_time
    # Tính độ chính xác
    correct = accuracy_score([label_predict], [input_label])
    if correct == 1.0:
        mAP = 1
    else: 
        mAP = 0            
    
    # in kết quả ra màn hình console
    print(f"Top-{top_k}:")
    print(f"  mAP: {mAP:.3f}")
    print(f"  Accuracy: {correct:.3f}")
    print(f"  Query Time: {query_time:.3f} seconds")
    print(f"predicted label: {label_predict}")
    
    top_k_image_paths = [database_images[i] for i in top_k_indices]
    
    # showImages(image_path, top_k_image_paths, top_k, image_label, label_predict)
    
    modified_images = [] # các ảnh sau khi đánh dấu sẽ được deploy
    
    for img_path in top_k_image_paths:
        img = Image.open(img_path)
        padding = 10
        img_with_border = ImageOps.expand(img, border=padding, fill=(255, 255, 255))
        draw = ImageDraw.Draw(img_with_border)
        
        outline_width = 2
        top_left = (outline_width//2, outline_width//2)
        bottom_right = (img_with_border.width - outline_width//2, img_with_border.height - outline_width//2)
        
        if image_label == label_predict:
            draw.rectangle([top_left, bottom_right], outline='blue', width=2)
        else:
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            
        modified_images.append(img_with_border)
        
    results = {
        'retrieval_results': modified_images,
        'accuracy': correct,
        'mAP': mAP, 
        'query_time': query_time,
        
    }
        
    return results
