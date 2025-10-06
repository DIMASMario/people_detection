import numpy as np
from scipy.spatial import distance
import time
import cv2
import datetime
from reid import PersonReID
import logging

class Tracker:
    def __init__(self, max_disappeared=40, max_distance=50, exit_timeout=10, 
                 reid_enabled=True, reid_threshold=0.6, reid_model='resnet50', device=''):
        """
        Inisialisasi tracker untuk melacak objek yang terdeteksi
        
        Args:
            max_disappeared: Jumlah frame maksimal objek dapat menghilang sebelum dihapus
            max_distance: Jarak maksimal (pixel) antara posisi objek di frame sebelumnya
                        dan frame saat ini untuk dianggap sebagai objek yang sama
            exit_timeout: Waktu (detik) setelah tidak terlihat untuk menandai objek sebagai keluar
            reid_enabled: Aktifkan fitur person re-identification
            reid_threshold: Ambang batas kesamaan untuk mengenali orang yang sama
            reid_model: Model untuk person re-identification
            device: Device untuk inferensi ('cpu', '0', dll.)
        """
        self.next_object_id = 0
        self.objects = {}  # Dictionary untuk menyimpan objek {ID: centroid}
        self.disappeared = {}  # Dictionary untuk menghitung frame saat objek menghilang
        self.positions = {}  # Dictionary untuk menyimpan posisi objek {ID: [x1, y1, x2, y2]}
        self.timestamps = {}  # Dictionary untuk mencatat timestamp {ID: {'first_seen', 'last_seen'}}
        self.states = {}  # Dictionary untuk menyimpan status objek {ID: 'new'/'scanned'/'exited'}
        self.entry_times = {}  # Dictionary untuk menyimpan waktu masuk {ID: datetime}
        self.exit_times = {}  # Dictionary untuk menyimpan waktu keluar {ID: datetime}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.exit_timeout = exit_timeout
        
        # Person ID mapping
        self.reid_enabled = reid_enabled
        self.person_ids = {}  # Dictionary untuk menyimpan person ID {track_id: person_id}
        self.counted_people = set()  # Set untuk menyimpan person ID yang sudah dihitung
        
        # Inisialisasi PersonReID jika diaktifkan
        if reid_enabled:
            try:
                self.reid = PersonReID(
                    model_type=reid_model,
                    similarity_threshold=reid_threshold,
                    device=device
                )
                logging.info(f"Person ReID diaktifkan dengan model {reid_model}")
            except Exception as e:
                logging.error(f"Error menginisialisasi PersonReID: {e}")
                self.reid_enabled = False
                logging.warning("Person ReID dinonaktifkan karena error")
        
    def register(self, centroid, bbox, frame=None):
        """Daftarkan objek baru"""
        # Tracking ID baru
        track_id = self.next_object_id
        
        # Simpan informasi tracking
        self.objects[track_id] = centroid
        self.positions[track_id] = bbox
        self.disappeared[track_id] = 0
        self.timestamps[track_id] = {
            'first_seen': datetime.datetime.now(),
            'last_seen': datetime.datetime.now()
        }
        self.states[track_id] = "new"
        
        # Person re-identification jika diaktifkan dan frame tersedia
        person_id = None
        if self.reid_enabled and frame is not None:
            person_id, is_new = self.reid.identify_person(frame, bbox, track_id)
            self.person_ids[track_id] = person_id
            
            # Jika person sudah pernah dihitung, tandai langsung sebagai 'counted'
            if person_id in self.counted_people:
                self.mark_as_counted(track_id)
                logging.info(f"Person ID {person_id} sudah pernah dihitung, track ID {track_id} ditandai sebagai counted")
        
        self.next_object_id += 1
        return track_id
    
    def deregister(self, object_id):
        """Hapus objek yang tidak lagi terlihat"""
        # Periksa apakah objek yang tidak terlihat lagi sudah melewati exit_timeout
        if self.states[object_id] == "scanned":
            # Tandai sebagai sudah keluar
            self.states[object_id] = "exited"
            self.exit_times[object_id] = datetime.datetime.now()
            
            # Jika person re-identification aktif, tandai person ID sebagai dihitung
            if self.reid_enabled and object_id in self.person_ids:
                self.counted_people.add(self.person_ids[object_id])
        
        # Hapus dari objek tracking aktif
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.positions[object_id]
    
    def mark_as_scanned(self, object_id):
        """
        Tandai objek sebagai sudah di-scan
        
        Args:
            object_id: ID objek yang dilacak
        """
        if object_id in self.objects:
            self.states[object_id] = "scanned"
            # Print debug info
            print(f"Tracker: Object {object_id} marked as scanned")
            
            # Update database atau sistem lain bisa dilakukan di sini
            return True
        return False
    
    def mark_as_counted(self, object_id):
        """Tandai objek sebagai sudah dihitung (untuk ReID)"""
        if object_id in self.states:
            # Update status jika belum counted
            if self.states[object_id] == "new":
                self.states[object_id] = "scanned"
                self.entry_times[object_id] = datetime.datetime.now()
    
    def is_already_counted(self, object_id):
        """Cek apakah objek sudah pernah dihitung sebelumnya"""
        if not self.reid_enabled:
            return False
            
        if object_id in self.person_ids:
            person_id = self.person_ids[object_id]
            return person_id in self.counted_people
            
        return False
    
    def update(self, detections, frame=None):
        """
        Perbarui posisi objek yang dilacak
        
        Args:
            detections: List deteksi [x1, y1, x2, y2, confidence]
            frame: Frame gambar untuk ekstraksi fitur ReID
            
        Returns:
            Dictionary objek yang dilacak {ID: (centroid, bbox, timestamp, state)}
        """
        print(f"Tracker: menerima {len(detections)} deteksi")
        
        # PERBAIKAN: Hapus objek terlama jika jumlah objek terlalu banyak
        if len(self.objects) > 2 * len(detections) and len(detections) > 0:
            print(f"Tracker: Terlalu banyak objek ({len(self.objects)}), membersihkan objek lama")
            # Ambil objek terlama berdasarkan disappeared count
            objects_to_remove = sorted(
                [(obj_id, disap) for obj_id, disap in self.disappeared.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Hapus setengah dari objek terlama
            for obj_id, _ in objects_to_remove[:len(objects_to_remove)//2]:
                self.deregister(obj_id)
            
            print(f"Tracker: Setelah dibersihkan, tersisa {len(self.objects)} objek")
        
        # Jika tidak ada deteksi, tandai semua objek yang dilacak sebagai menghilang
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Jika objek telah menghilang terlalu lama, deregister
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.get_objects()
        
        # Inisialisasi array untuk centroid saat ini
        input_centroids = np.zeros((len(detections), 2), dtype="float")
        input_bboxes = []
        
        # Hitung centroid untuk setiap deteksi
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, _ = detection
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_centroids[i] = (cx, cy)
            input_bboxes.append((x1, y1, x2, y2))
        
        # Jika tidak ada objek yang dilacak, daftarkan semua deteksi
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], frame)
        
        # Jika tidak, coba cocokkan deteksi baru dengan objek yang ada
        else:
            # Ambil ID objek dan centroid yang ada
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Hitung jarak antara setiap pasangan centroid objek dan input
            from scipy.spatial import distance
            D = distance.cdist(np.array(object_centroids), input_centroids)
            
            # Cari jarak terkecil untuk setiap baris, lalu urutkan indeks
            rows = D.min(axis=1).argsort()
            
            # Cari jarak terkecil untuk setiap kolom, lalu urutkan indeks
            cols = D.argmin(axis=1)[rows]
            
            # Simpan baris dan kolom yang telah diproses
            used_rows = set()
            used_cols = set()
            
            # Loop over kombinasi indeks baris dan kolom
            for (row, col) in zip(rows, cols):
                # Jika sudah diproses, skip
                if row in used_rows or col in used_cols:
                    continue
                
                # Jika jarak terlalu jauh, skip
                if D[row, col] > self.max_distance:
                    continue
                
                # Dapatkan ID objek untuk baris saat ini, tetapkan centroid baru
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.positions[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0
                self.timestamps[object_id]['last_seen'] = datetime.datetime.now()
                
                # Tandai baris dan kolom sebagai digunakan
                used_rows.add(row)
                used_cols.add(col)
            
            # Proses baris yang tidak digunakan
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                # Ambil ID objek dan tingkatkan penghitung disappeared
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                # Deregister jika telah menghilang terlalu lama
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Daftarkan deteksi baru yang tidak digunakan
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col], frame)
        
        # Debugging
        print(f"Tracker: menghasilkan {len(self.objects)} objek terlacak")
        
        return self.get_objects()
    
    def get_objects(self):
        """
        Returns:
            Dictionary objek yang dilacak {ID: (centroid, bbox, timestamp, state)}
        """
        result = {}
        for object_id in self.objects:
            entry_time = self.entry_times.get(object_id)
            exit_time = self.exit_times.get(object_id)
            
            # Tambahkan person_id jika tersedia
            person_id = None
            if self.reid_enabled and object_id in self.person_ids:
                person_id = self.person_ids[object_id]
                
            result[object_id] = {
                'centroid': self.objects[object_id],
                'bbox': self.positions[object_id],
                'timestamps': self.timestamps[object_id],
                'state': self.states[object_id],
                'entry_time': entry_time,
                'exit_time': exit_time,
                'person_id': person_id,
                'already_counted': self.is_already_counted(object_id)
            }
        return result
        
    def check_for_exits(self):
        """
        Periksa objek yang tidak terlihat untuk waktu tertentu dan tandai sebagai keluar
        
        Returns:
            List ID objek yang telah keluar
        """
        exited_ids = []
        current_time = datetime.datetime.now()
        
        for object_id, timestamp in list(self.timestamps.items()):
            # Hanya periksa objek yang sudah di-scan (melewati garis)
            if object_id in self.states and self.states[object_id] == "scanned":
                last_seen = timestamp['last_seen']
                time_diff = (current_time - last_seen).total_seconds()
                
                # Jika melebihi exit_timeout, tandai sebagai keluar
                if time_diff > self.exit_timeout and object_id in self.objects:
                    self.states[object_id] = "exited"
                    self.exit_times[object_id] = current_time
                    exited_ids.append(object_id)
                    self.deregister(object_id)
        
        return exited_ids