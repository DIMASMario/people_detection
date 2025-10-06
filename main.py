import os
import time
import yaml
import cv2
import argparse
import logging
from dotenv import load_dotenv
from detector import PersonDetector
from tracker import Tracker
from counter import VisitorCounter
from visualizer import Visualizer
from exporter import DataExporter
from database import Database
from api import initialize as init_api, start_api_server
import signal
import sys
import threading

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visitor_counter.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Sistem Pendeteksi Pengunjung dengan YOLOv5")
    parser.add_argument('--config', type=str, default='config.yaml', help='path ke file konfigurasi')
    parser.add_argument('--source', type=str, default=None, help='sumber video (0 untuk webcam, atau path ke file video)')
    parser.add_argument('--conf-thres', type=float, default=None, help='confidence threshold')
    parser.add_argument('--output', type=str, default=None, help='path untuk menyimpan hasil')
    parser.add_argument('--db', type=str, default='visitors.db', help='path database SQLite')
    parser.add_argument('--port', type=int, default=None, help='port untuk API')
    return parser.parse_args()

def main():
    # Muat variabel lingkungan
    load_dotenv()
    
    # Parsing argumen
    args = parse_args()
    
    # Muat konfigurasi
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override konfigurasi dengan variabel lingkungan dan argumen
    # Prioritas: argumen > env vars > config file
    if args.source:
        config['camera']['source'] = args.source
    elif os.getenv('RTSP_URL'):
        config['camera']['source'] = os.getenv('RTSP_URL')
        
    if args.conf_thres:
        config['detection']['conf_threshold'] = args.conf_thres
        
    if args.output:
        config['export']['excel_path'] = args.output
        
    if args.port:
        config['api']['port'] = args.port
    elif os.getenv('PORT'):
        config['api']['port'] = int(os.getenv('PORT'))
        
    if os.getenv('EXIT_TIMEOUT'):
        config['tracking']['exit_timeout'] = int(os.getenv('EXIT_TIMEOUT'))
    
    # Inisialisasi database
    db = Database(args.db)
    logging.info(f"Database diinisialisasi: {args.db}")
    
    # Inisialisasi kamera
    source = config['camera']['source']
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    logging.info(f"Mencoba membuka sumber video: {source}")
    cap = cv2.VideoCapture(source)
    
    # Periksa apakah kamera berhasil dibuka
    if not cap.isOpened():
        logging.error(f"Gagal membuka sumber video: {source}")
        print(f"ERROR: Tidak dapat membuka sumber video '{source}'. Pastikan webcam terhubung atau URL RTSP benar.")
        return
    
    # Set properti kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
    
    # Baca frame pertama untuk memastikan koneksi berhasil
    ret, test_frame = cap.read()
    if not ret:
        logging.error("Tidak dapat membaca frame dari sumber video")
        print("ERROR: Tidak dapat membaca frame dari sumber video. Pastikan webcam atau RTSP stream berfungsi.")
        return
    
    logging.info(f"Berhasil terhubung ke sumber video. Resolusi: {test_frame.shape[1]}x{test_frame.shape[0]}")
    
    # Inisialisasi komponen
    detector = PersonDetector(
        model_size=config['detection']['model_size'],
        conf_threshold=config['detection']['conf_threshold'],
        iou_threshold=config['detection']['iou_threshold'],
        classes=config['detection']['classes'],
        device=config['detection']['device']
    )
    
    # Inisialisasi tracker dengan person re-identification
    tracker = Tracker(
        max_disappeared=config['tracking']['max_disappeared'],
        max_distance=config['tracking']['max_distance'],
        exit_timeout=config['tracking']['exit_timeout'],
        reid_enabled=config.get('reid', {}).get('enabled', True),
        reid_threshold=config.get('reid', {}).get('similarity_threshold', 0.6),
        reid_model=config.get('reid', {}).get('model', 'resnet50'),
        device=config.get('reid', {}).get('device', '')
    )
    
    counter = VisitorCounter(
        line_position=config['counting_line']['position'],
        direction=config['counting_line']['direction'],
        frame_width=config['camera']['width'],  # Garis vertikal menggunakan frame width
        db=db  # Tambahkan database instance
    )
    
    visualizer = Visualizer()
    
    exporter = DataExporter(
        db=db,
        excel_path=config['export']['excel_path'],
        auto_save_interval=config['export']['auto_save_interval']
    )
    
    # Inisialisasi API jika diaktifkan
    if config['api']['enabled']:
        init_api(db, counter, exporter)
        api_thread = start_api_server(config['api']['port'])
        logging.info(f"API server dijalankan di port {config['api']['port']}")
    
    logging.info(f"Memulai sistem penghitung pengunjung...")
    logging.info(f"Sumber video: {'Webcam' if source == 0 else source}")
    logging.info(f"Model YOLOv5{config['detection']['model_size']} dimuat")
    logging.info(f"Tekan 'q' untuk keluar")
    
    # Tambahkan handler untuk keyboard interrupt dan keluar
    def signal_handler(sig, frame):
        global exporter, cap
        logging.info('Menghentikan program...')
        print('\nProgram dihentikan')
        
        # Ekspor data sebelum keluar
        if 'exporter' in globals() and exporter:
            print("Mengekspor data ke Excel...")
            try:
                count = exporter.export_to_excel()
                print(f"Data berhasil diekspor: {count} record")
            except Exception as e:
                print(f"Gagal mengekspor data: {e}")
        
        # Release resources
        if 'cap' in globals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Set fullscreen mode
    cv2.namedWindow("Pendeteksi Pengunjung", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Pendeteksi Pengunjung", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Inisialisasi variabel
    objects = {}
    last_check_exit_time = time.time()
    last_save_time = time.time()
    prev_time = time.time()
    process_this_frame = True
    detections = []  # Inisialisasi detections sebagai list kosong

    # Loop utama
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Tidak dapat membaca frame. Keluar...")
            break
        
        # FPS throttling untuk performa lebih baik
        current_time = time.time()
        elapsed_time = current_time - prev_time
        
        # Hanya proses frame jika waktunya cukup (untuk mencapai target FPS)
        if elapsed_time > 1.0/config['camera']['fps']:
            process_this_frame = True
            prev_time = current_time
        else:
            process_this_frame = False
        
        # Process frame jika waktunya tepat
        if process_this_frame:
            # Deteksi orang
            detections = detector.detect(frame)
            
            # Debug - tampilkan informasi deteksi
            print(f"Main: Deteksi orang: {len(detections)}")
            
            # Visualisasi raw detections
            raw_vis = frame.copy()
            for det in detections:
                x1, y1, x2, y2, conf = det
                # Tampilkan semua deteksi dengan warna merah
                cv2.rectangle(raw_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(raw_vis, f"{conf:.2f}", (int(x1), int(y1) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Tampilkan raw detections
            cv2.imshow("Raw Detections", raw_vis)
            
            # Update tracker dengan deteksi baru
            objects = tracker.update(detections, frame if config['reid']['enabled'] else None)
            
            # Pelacakan objek
            objects = tracker.update(detections, frame if config['reid']['enabled'] else None)
            
            # Periksa objek yang keluar (exit) secara berkala
            if current_time - last_check_exit_time >= 1.0:
                exited_ids = tracker.check_for_exits()
                last_check_exit_time = current_time
            
            # Hitung pengunjung yang melewati garis
            if config['counting_line']['enabled']:
                prev_count = counter.count
                current_count = counter.count_visitors(objects, tracker)
                
                # Jika jumlah berubah, log perubahan
                if current_count != prev_count:
                    logging.info(f"Jumlah pengunjung berubah: {prev_count} -> {current_count}")
        
        # Selalu visualisasikan hasilnya
        output_frame = visualizer.draw_results(
            frame, 
            objects,  # Sekarang objects sudah pasti terdefinisi
            counter.count,
            counter.line_position if config['counting_line']['enabled'] else None
        )
        
        # Tampilkan
        cv2.imshow("Pendeteksi Pengunjung", output_frame)
        
        # Auto save
        if config['export']['enabled'] and (current_time - last_save_time) >= config['export']['auto_save_interval']:
            exporter.export_to_excel()
            last_save_time = current_time
            logging.info(f"Data otomatis disimpan ke {config['export']['excel_path']}")
        
        # Periksa jika pengguna menekan 'q'
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            logging.info("Program dihentikan oleh pengguna")
            
            # Ekspor data sebelum keluar
            if config['export']['enabled']:
                print("Mengekspor data ke Excel...")
                count = exporter.export_to_excel()
                print(f"Data berhasil diekspor: {count} record")
            
            break
    
    # Bersihkan
    cap.release()
    cv2.destroyAllWindows()
    
    # Ekspor data terakhir jika diaktifkan
    if config['export']['enabled']:
        exporter.export_to_excel()
        logging.info(f"Data terakhir disimpan ke {config['export']['excel_path']}")
    
    # Pastikan data diekspor saat keluar
    if config['export']['enabled']:
        print("Mengekspor data ke Excel...")
        try:
            count = exporter.export_to_excel()
            print(f"Data berhasil diekspor: {count} record")
        except Exception as e:
            print(f"Error saat ekspor ke Excel: {e}")
            print("Menyimpan data ke file alternatif...")
            
            # Gunakan nama file dengan timestamp sebagai alternatif
            import datetime
            timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            alt_excel_path = f"hasil_deteksi_pengunjung_{timestamp_str}.xlsx"
            
            try:
                # Gunakan metode langsung ke database untuk ekspor
                visitors = db.get_all_visitors()
                if visitors:
                    import pandas as pd
                    data = []
                    for visitor in visitors:
                        # Ambil data dari record database
                        visitor_id = visitor[0]
                        timestamp = visitor[1]
                        direction = visitor[2] if len(visitor) > 2 else "unknown"
                        notes = visitor[3] if len(visitor) > 3 else ""
                        
                        # Convert timestamp UNIX ke datetime yang dapat dibaca
                        visit_datetime = datetime.datetime.fromtimestamp(timestamp)
                        formatted_datetime = visit_datetime.strftime('%Y-%m-%d %H:%M:%S')
                        
                        data.append({
                            'id': visitor_id,
                            'timestamp': timestamp,
                            'datetime': formatted_datetime,
                            'direction': direction,
                            'notes': notes
                        })
                    
                    df = pd.DataFrame(data)
                    df.to_excel(alt_excel_path, index=False)
                    print(f"Data berhasil disimpan ke {alt_excel_path}")
            except Exception as backup_error:
                print(f"Gagal menyimpan data ke file alternatif: {backup_error}")
                print("Data tetap tersimpan di database dan dapat diekspor nanti")

if __name__ == "__main__":
    main()