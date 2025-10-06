import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import time
import logging

class PersonReID:
    """Kelas untuk person re-identification menggunakan deep features"""
    
    def __init__(self, model_type='resnet50', similarity_threshold=0.6, device=''):
        """
        Inisialisasi model re-identification
        
        Args:
            model_type: Tipe model yang digunakan (resnet18, resnet50, etc.)
            similarity_threshold: Threshold untuk menentukan kecocokan (0-1)
            device: Device untuk inferensi ('cpu', '0', dll.)
        """
        self.similarity_threshold = similarity_threshold
        self.gallery = {}  # Menyimpan fitur orang yang sudah dihitung
        self.person_count = 0  # Jumlah orang unik yang terdeteksi
        self.person_id_mapping = {}  # Mapping dari tracking_id ke person_id
        
        # Tentukan device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logging.info(f"Inisialisasi PersonReID dengan model {model_type} pada {self.device}")
        
        # Load model
        try:
            # Gunakan model pre-trained dari torchvision
            if model_type == 'resnet18':
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
                self.feature_dim = 512
            elif model_type == 'resnet50':
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
                self.feature_dim = 2048
            elif model_type == 'mobilenet_v2':
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
                self.feature_dim = 1280
            else:
                # Default ke resnet50
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
                self.feature_dim = 2048
            
            # Hapus layer terakhir (classifier) untuk mendapatkan fitur
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logging.info(f"Model PersonReID berhasil dimuat")
        except Exception as e:
            logging.error(f"Error memuat model PersonReID: {e}")
            # Fallback ke ekstraksi fitur sederhana jika model gagal dimuat
            self.model = None
            logging.warning("Menggunakan ekstraksi fitur histogram sebagai fallback")
        
        # Transformasi untuk preprocessing gambar
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, frame, bbox):
        """
        Ekstrak fitur dari person detection
        
        Args:
            frame: Frame gambar
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Feature vector untuk identifikasi
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Pastikan koordinat dalam batasan frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        if x2 <= x1 or y2 <= y1:
            # Kembalikan fitur kosong jika bbox tidak valid
            return np.zeros(self.feature_dim if self.model else 64, dtype=np.float32)
        
        # Crop bagian pengunjung
        person_img = frame[y1:y2, x1:x2]
        
        # Jika model deep learning tersedia, gunakan
        if self.model is not None:
            try:
                # Convert OpenCV BGR ke RGB
                person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                # Convert ke PIL Image
                pil_img = Image.fromarray(person_img_rgb)
                # Apply transform
                img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                # Extract feature dengan model
                with torch.no_grad():
                    features = self.model(img_tensor)
                    features = features.squeeze().cpu().numpy()
                
                # Normalize fitur
                features = features / np.linalg.norm(features)
                return features
            except Exception as e:
                logging.error(f"Error ekstraksi fitur deep: {e}")
                # Fallback ke histogram jika ekstraksi gagal
                return self._extract_histogram_features(person_img)
        else:
            # Gunakan histogram sebagai fallback
            return self._extract_histogram_features(person_img)
    
    def _extract_histogram_features(self, person_img):
        """Ekstrak fitur histogram sebagai fallback"""
        # Resize ke ukuran standar
        person_img = cv2.resize(person_img, (64, 128))
        
        # Ekstrak fitur sederhana (histogram warna)
        hist_features = []
        for i in range(3):  # untuk 3 channel (BGR)
            hist = cv2.calcHist([person_img], [i], None, [8], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        # Konversi ke array numpy
        features = np.array(hist_features, dtype=np.float32)
        # Normalize fitur
        features = features / np.linalg.norm(features)
        return features
    
    def compare_features(self, features1, features2):
        """
        Bandingkan dua set fitur dan kembalikan skor kesamaan (0-1)
        """
        # Hitung cosine similarity
        similarity = np.dot(features1, features2)
        return float(similarity)
    
    def identify_person(self, frame, bbox, tracking_id):
        """
        Identifikasi apakah orang ini sudah pernah dilihat
        
        Args:
            frame: Frame gambar
            bbox: Bounding box [x1, y1, x2, y2]
            tracking_id: ID tracking dari sistem pelacakan
            
        Returns:
            (person_id, is_new): ID unik orang dan flag apakah orang baru
        """
        # Jika tracking_id sudah ada di mapping, gunakan person_id yang sama
        if tracking_id in self.person_id_mapping:
            return self.person_id_mapping[tracking_id], False
        
        # Ekstrak fitur
        features = self.extract_features(frame, bbox)
        
        # Bandingkan dengan gallery
        best_match_id = None
        best_match_score = 0
        
        for person_id, stored_features in self.gallery.items():
            similarity = self.compare_features(features, stored_features)
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_id = person_id
        
        # Jika kecocokan ditemukan di atas threshold
        if best_match_id is not None and best_match_score > self.similarity_threshold:
            # Perbarui mapping
            self.person_id_mapping[tracking_id] = best_match_id
            return best_match_id, False
        
        # Jika tidak ditemukan kecocokan, buat ID baru
        self.person_count += 1
        new_person_id = self.person_count
        
        # Simpan fitur ke gallery
        self.gallery[new_person_id] = features
        
        # Perbarui mapping
        self.person_id_mapping[tracking_id] = new_person_id
        
        return new_person_id, True
    
    def update_gallery(self, person_id, frame, bbox):
        """
        Perbarui fitur di gallery untuk person_id tertentu
        
        Args:
            person_id: ID unik orang
            frame: Frame gambar
            bbox: Bounding box [x1, y1, x2, y2]
        """
        if person_id in self.gallery:
            # Ekstrak fitur baru
            new_features = self.extract_features(frame, bbox)
            
            # Update gallery dengan rata-rata berbobot
            # 70% fitur lama, 30% fitur baru untuk adaptasi gradual
            self.gallery[person_id] = 0.7 * self.gallery[person_id] + 0.3 * new_features
            
            # Normalize hasil
            self.gallery[person_id] = self.gallery[person_id] / np.linalg.norm(self.gallery[person_id])
    
    # Tambahkan fungsi untuk meningkatkan performa
    def preprocess_image(self, person_img, enhance_contrast=True):
        """Preprocessing gambar untuk meningkatkan kualitas fitur"""
        # Resize ke ukuran standar
        person_img = cv2.resize(person_img, (128, 256))
        
        # Opsional: tingkatkan kontras untuk fitur yang lebih jelas
        if enhance_contrast:
            lab = cv2.cvtColor(person_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            person_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return person_img