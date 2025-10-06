import numpy as np
import datetime
import logging

class VisitorCounter:
    def __init__(self, line_position=0.5, direction="left", frame_width=640, db=None):
        """
        Inisialisasi penghitung pengunjung dengan garis vertikal
        
        Args:
            line_position: Posisi garis virtual relatif terhadap lebar frame (0-1)
            direction: Arah penghitungan - diubah ke "left" sebagai default
            frame_width: Lebar frame video
            db: Database instance
        """
        self.line_position = int(line_position * frame_width)
        self.direction = "left"  # Hanya menghitung arah kanan ke kiri
        self.count = 0
        self.total_left = 0  # Bergerak dari kanan ke kiri
        self.total_right = 0  # Tidak akan dihitung
        
        # Dictionary untuk menyimpan status terakhir objek
        self.last_positions = {}  # {object_id: {"position": "left"/"right", "counted": False}}
        
        # Set untuk menyimpan ID objek yang sudah dihitung
        self.counted_ids = set()
        
        # Database reference
        self.db = db
        
        # Debug mode
        self.debug = True
    
    def count_visitors(self, objects, tracker):
        """
        Hitung pengunjung yang melintasi garis vertikal - HANYA DARI KANAN KE KIRI
        """
        # Jika tidak ada objek, langsung return
        if not objects:
            return self.count
        
        print(f"Counter: Checking {len(objects)} objects")
        
        for object_id, data in objects.items():
            # Jika objek sudah dihitung, lewati
            if object_id in self.counted_ids:
                continue
                
            centroid = data['centroid']
            cx = centroid[0]  # Koordinat X untuk garis vertikal
            state = data.get('state', 'new')  # Default ke 'new' jika tidak ada
            
            # Tentukan posisi saat ini
            current_position = "left" if cx < self.line_position else "right"
            
            # Inisialisasi objek baru jika belum ada di last_positions
            if object_id not in self.last_positions:
                self.last_positions[object_id] = {
                    "position": current_position,
                    "counted": False
                }
                continue
            
            # Ambil data terakhir
            last_data = self.last_positions[object_id]
            last_position = last_data["position"]
            already_counted = last_data["counted"]
            
            # Cek apakah objek melintasi garis DARI KANAN KE KIRI
            if not already_counted and object_id not in self.counted_ids:
                if last_position == "right" and current_position == "left":
                    print(f"Counter: Object {object_id} counted as visitor (crossing line RIGHT TO LEFT)")
                    self.count += 1
                    self.total_left += 1
                    
                    # Tandai sebagai sudah dihitung
                    self.last_positions[object_id]["counted"] = True
                    self.counted_ids.add(object_id)
                    
                    # Tandai di tracker
                    tracker.mark_as_scanned(object_id)
                    
                    # Tambahkan ke database dengan waktu REALTIME saat ini
                    if self.db:
                        current_time = datetime.datetime.now().timestamp()
                        visitor_id = self.db.add_visitor(
                            timestamp=current_time,
                            direction="left",
                            notes=f"Object ID: {object_id} - Right to Left"
                        )
                        print(f"Added to database with ID: {visitor_id} at {datetime.datetime.now()}")
                    
                    # Log
                    logging.info(f"Pengunjung terdeteksi (kanan ke kiri)! ID: {object_id}, Total: {self.count}")
            
            # Update posisi terakhir
            self.last_positions[object_id]["position"] = current_position
        
        return self.count
    
    def get_stats(self):
        """
        Dapatkan statistik penghitungan
        
        Returns:
            Dictionary dengan statistik penghitungan
        """
        return {
            "total": self.count,
            "total_left": self.total_left,
            "total_right": self.total_right
        }