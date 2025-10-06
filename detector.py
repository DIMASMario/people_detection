import os
import torch
import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_size='n', conf_threshold=0.4, iou_threshold=0.5, classes=None, device=''):
        """
        Inisialisasi detektor orang dengan YOLOv5
        
        Args:
            model_size: Ukuran model YOLOv5 ('n', 's', 'm', 'l', 'x')
            conf_threshold: Confidence threshold (0-1)
            iou_threshold: IoU threshold untuk NMS (0-1)
            classes: List class index untuk dideteksi (None untuk semua class)
            device: Device untuk inferensi ('cpu', '0', dll.)
        """
        # Turunkan threshold untuk deteksi lebih banyak
        self.model_size = model_size
        self.conf_threshold = conf_threshold  # Turunkan ke 0.4 dari 0.7
        self.iou_threshold = iou_threshold
        self.classes = classes if classes is not None else [0]  # Hanya person
        self.skip_frames = 0  # Jangan skip frame agar deteksi lebih konsisten
        self.current_frame = 0
        self.last_detections = []  # Simpan deteksi terakhir
        
        # Tentukan device
        if device:
            self.device = device
        else:
            self.device = 'cpu'  # Default ke CPU
        
        try:
            # Coba gunakan model lokal terlebih dahulu
            local_models = [
                os.path.join(os.path.dirname(__file__), 'yolov5nu.pt'),
                os.path.join(os.path.dirname(__file__), '..', 'yolov5', f'yolov5{model_size}.pt'),
                os.path.join(os.path.dirname(__file__), '..', 'yolov5', 'yolov5s.pt')  # Default yang biasanya ada
            ]
            
            model_loaded = False
            for model_path in local_models:
                if os.path.exists(model_path):
                    print(f"[INFO] Mencoba memuat model dari {model_path}")
                    from ultralytics import YOLO
                    self.model = YOLO(model_path)
                    print(f"[INFO] Berhasil memuat model dari {model_path}")
                    self.model_type = "yolov5"
                    model_loaded = True
                    break
                    
            if not model_loaded:
                raise FileNotFoundError("Tidak menemukan model YOLOv5 lokal")
            
        except Exception as e:
            print(f"[WARNING] Gagal memuat YOLOv5: {e}")
            print(f"[INFO] Menggunakan model fallback...")
            
            # Import torchvision sebagai fallback
            import torchvision
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device).eval()
            self.model_type = "fasterrcnn"
            print(f"[INFO] Model fallback FasterRCNN dimuat di {self.device}")
    
    def detect(self, frame):
        """
        Deteksi orang dalam frame
        
        Args:
            frame: Frame gambar untuk dideteksi
            
        Returns:
            List deteksi: [x1, y1, x2, y2, confidence]
        """
        try:
            # Debug info - tampilkan frame shape
            h, w = frame.shape[:2]
            print(f"Frame shape: {w}x{h}")
            
            if self.model_type == "yolov5":
                # Ultralytics API
                results = self.model.predict(
                    frame, 
                    conf=self.conf_threshold,  # Threshold yang lebih rendah
                    iou=self.iou_threshold, 
                    classes=self.classes,  # Class 0 adalah 'person' di COCO
                    verbose=False
                )
                
                # Konversi hasil ke format yang diperlukan
                detections = []
                if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    # LEBIH PERMISIF - longgarkan filter
                    min_width = 30  # Turunkan dari 50
                    min_height = 60  # Turunkan dari 100
                    
                    print(f"Jumlah deteksi awal: {len(results[0].boxes)}")
                    
                    for result in results[0].boxes.data.cpu().numpy():
                        x1, y1, x2, y2, conf, cls = result
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Filter berdasarkan ukuran minimum - LEBIH PERMISIF
                        if width > min_width and height > min_height:
                            detections.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
                    
                    print(f"Jumlah deteksi setelah filter: {len(detections)}")
                    return detections
                else:
                    print("Tidak ada deteksi dari model YOLOv5")
                    return []
                
            else:
                # Fallback ke deteksi kosong
                print("Model tidak dikenali")
                return []
                
        except Exception as e:
            print(f"[ERROR] Deteksi gagal: {e}")
            import traceback
            traceback.print_exc()
            return []