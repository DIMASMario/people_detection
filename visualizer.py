import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        """
        Inisialisasi visualizer untuk menampilkan hasil deteksi dan pelacakan
        """
        # Warna untuk berbagai elemen visual (BGR format)
        self.colors = {
            'new': (0, 255, 0),       # Hijau untuk pengunjung baru
            'scanned': (0, 165, 255),  # Oranye untuk pengunjung yang sedang dihitung
            'exited': (0, 0, 255),    # Merah untuk pengunjung yang keluar
            'line': (255, 0, 0),      # Biru untuk garis
            'text': (255, 255, 255),  # Putih untuk teks
            'id': (255, 255, 0)       # Cyan untuk ID
        }
    
    def draw_results(self, frame, objects, count, line_position=None):
        """
        Gambar hasil deteksi, tracking, dan penghitungan pada frame
        
        Args:
            frame: Frame video untuk digambar
            objects: Dictionary objek yang dilacak
            count: Jumlah total pengunjung
            line_position: Posisi garis penghitungan vertikal (opsional)
            
        Returns:
            Frame yang telah digambar dengan visualisasi
        """
        # Debug info
        print(f"Visualizer: objects={len(objects)}")
        
        # Buat salinan frame untuk digambar
        output = frame.copy()
        
        # Gambar garis penghitungan vertikal jika ada
        if line_position is not None:
            cv2.line(output, (line_position, 0), (line_position, output.shape[0]), 
                     self.colors['line'], 2)
            cv2.putText(output, "Garis Penghitungan", (line_position + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Hitung jumlah pengunjung berdasarkan status
        counts = {'new': 0, 'scanned': 0, 'exited': 0}
        
        # TAMBAHKAN INI: Cek apakah ada objek yang terlacak
        if len(objects) == 0:
            cv2.putText(output, "TIDAK ADA OBJEK TERLACAK", (10, output.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Gambar setiap objek yang dilacak
        for object_id, data in objects.items():
            # Ekstrak bbox dan centroid
            bbox = data['bbox']
            centroid = data['centroid']
            state = data['state']
            
            # Update penghitungan
            counts[state] = counts.get(state, 0) + 1
            
            # Tentukan warna box berdasarkan status
            box_color = self.colors.get(state, self.colors['new'])
            
            # Gambar bounding box
            cv2.rectangle(output, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                         box_color, 2)
            
            # Gambar ID objek
            text = f"ID: {object_id} [{state}]"
            cv2.putText(output, text, (int(bbox[0]), int(bbox[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['id'], 2)
            
            # Gambar centroid
            cv2.circle(output, (int(centroid[0]), int(centroid[1])), 4, self.colors['id'], -1)
        
        # Gambar jumlah pengunjung
        cv2.rectangle(output, (10, 10), (280, 130), (0, 0, 0), -1)
        cv2.putText(output, f"Total Pengunjung: {count}", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        cv2.putText(output, f"Objek Terlacak: {len(objects)}", (15, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        cv2.putText(output, f"Baru: {counts['new']} | Scanned: {counts['scanned']}", (15, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Tambahkan informasi arah pada overlay
        cv2.putText(output, "HANYA HITUNG: KANAN KE KIRI", (15, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return output