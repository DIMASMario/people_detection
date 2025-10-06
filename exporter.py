import pandas as pd
import os
import logging
import datetime
import time
import traceback

class DataExporter:
    def __init__(self, db, excel_path="hasil_deteksi_pengunjung.xlsx", auto_save_interval=300):
        """
        Inisialisasi exporter data pengunjung
        
        Args:
            db: Database instance
            excel_path: Path file Excel untuk menyimpan hasil
            auto_save_interval: Interval auto save dalam detik
        """
        self.db = db
        self.base_excel_path = excel_path  # Simpan path dasar
        self.excel_path = excel_path
        self.auto_save_interval = auto_save_interval
        self.last_export_id = 0
    
    def export_to_excel(self):
        """
        Export data pengunjung dari database ke file Excel dengan waktu REALTIME
        
        Returns:
            Jumlah record yang diekspor
        """
        try:
            # Dapatkan semua data pengunjung
            visitors = self.db.get_visitors(after_id=self.last_export_id)
            
            if not visitors:
                print("[INFO] Tidak ada data baru untuk diekspor")
                
                # Force re-export semua data jika tidak ada data baru
                all_visitors = self.db.get_all_visitors()
                if all_visitors:
                    visitors = all_visitors
                else:
                    return 0
            
            # Convert ke DataFrame untuk ekspor mudah ke Excel
            data = []
            for visitor in visitors:
                # Ambil data dari record database
                visitor_id = visitor[0]
                timestamp = visitor[1]  # Ini adalah UNIX timestamp
                direction = visitor[2] if len(visitor) > 2 else "unknown"
                notes = visitor[3] if len(visitor) > 3 else ""
                
                # Convert timestamp UNIX ke datetime yang dapat dibaca
                visit_datetime = datetime.datetime.fromtimestamp(timestamp)
                formatted_datetime = visit_datetime.strftime('%Y-%m-%d %H:%M:%S')
                
                data.append({
                    'id': visitor_id,
                    'timestamp': timestamp,
                    'datetime': formatted_datetime,  # Format waktu yang bisa dibaca
                    'direction': direction,
                    'notes': notes
                })
            
            df = pd.DataFrame(data)
            
            # Coba beberapa kali dengan nama file berbeda jika file asli tidak bisa diakses
            max_attempts = 3
            attempt = 0
            success = False
            
            while attempt < max_attempts and not success:
                try:
                    # Gunakan nama file dengan timestamp jika percobaan > 0
                    if attempt > 0:
                        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        file_name, file_ext = os.path.splitext(self.base_excel_path)
                        self.excel_path = f"{file_name}_{timestamp_str}{file_ext}"
                        print(f"Mencoba dengan nama file baru: {self.excel_path}")
                    
                    # Cek apakah file ada dan coba membukanya untuk memastikan tidak digunakan
                    if os.path.exists(self.excel_path):
                        try:
                            # Coba membuka dan menutupnya dengan mode 'a+' untuk memastikan dapat ditulis
                            with open(self.excel_path, 'a+') as test_file:
                                pass
                        except PermissionError:
                            # Jika error, artinya file masih digunakan
                            raise PermissionError(f"File {self.excel_path} sedang digunakan aplikasi lain")
                    
                    # Simpan ke Excel
                    df.to_excel(self.excel_path, index=False)
                    success = True
                    
                    # Update last_export_id
                    if visitors:
                        self.last_export_id = max(visitor[0] for visitor in visitors)
                    
                    logging.info(f"Data berhasil disimpan ke {self.excel_path}")
                
                except PermissionError as e:
                    attempt += 1
                    print(f"Percobaan {attempt}: Gagal mengakses file Excel - {str(e)}")
                    if attempt >= max_attempts:
                        raise
                    time.sleep(1)  # Tunggu sebentar sebelum mencoba lagi
            
            return len(visitors)
            
        except Exception as e:
            logging.error(f"Error saat ekspor ke Excel: {e}")
            traceback.print_exc()
            return 0