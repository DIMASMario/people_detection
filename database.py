from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os
import sqlite3
import time
import logging

Base = declarative_base()

class Visitor(Base):
    """Model tabel untuk visitor di database"""
    __tablename__ = 'visitors'
    
    id = Column(Integer, primary_key=True)
    track_id = Column(Integer)
    state = Column(String)
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    entry_time = Column(DateTime, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float)
    last_x = Column(Float)
    last_y = Column(Float)
    exported = Column(Boolean, default=False)
    
    def to_dict(self):
        return {
            "id": self.id,
            "track_id": self.track_id,
            "state": self.state,
            "first_seen": self.first_seen.strftime("%Y-%m-%d %H:%M:%S") if self.first_seen else None,
            "last_seen": self.last_seen.strftime("%Y-%m-%d %H:%M:%S") if self.last_seen else None,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M:%S") if self.entry_time else None,
            "exit_time": self.exit_time.strftime("%Y-%m-%d %H:%M:%S") if self.exit_time else None,
            "duration_seconds": self.duration_seconds,
            "last_x": self.last_x,
            "last_y": self.last_y,
            "exported": self.exported
        }

class Database:
    def __init__(self, db_path='visitors.db'):
        """
        Inisialisasi database untuk menyimpan data pengunjung
        
        Args:
            db_path: Path ke file database SQLite
        """
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """
        Inisialisasi struktur database
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Buat tabel visitors jika belum ada
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            direction TEXT,
            notes TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_visitor(self, timestamp=None, direction=None, notes=None):
        """
        Tambahkan pengunjung baru ke database
        
        Args:
            timestamp: Waktu deteksi (default: waktu saat ini)
            direction: Arah pengunjung (masuk/keluar)
            notes: Catatan tambahan
        
        Returns:
            ID pengunjung yang baru ditambahkan
        """
        if timestamp is None:
            timestamp = time.time()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO visitors (timestamp, direction, notes) VALUES (?, ?, ?)',
                (timestamp, direction, notes)
            )
            
            visitor_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return visitor_id
        except Exception as e:
            logging.error(f"Error saat menambahkan pengunjung ke database: {e}")
            return None
    
    def get_visitors(self, after_id=0, limit=None):
        """
        Ambil data pengunjung dari database
        
        Args:
            after_id: Hanya ambil data dengan ID lebih besar dari ini
            limit: Batasi jumlah record (None untuk tanpa batas)
            
        Returns:
            List data pengunjung [(id, timestamp, direction, notes), ...]
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if limit:
                cursor.execute('SELECT * FROM visitors WHERE id > ? ORDER BY id ASC LIMIT ?', (after_id, limit))
            else:
                cursor.execute('SELECT * FROM visitors WHERE id > ? ORDER BY id ASC', (after_id,))
                
            visitors = cursor.fetchall()
            conn.close()
            
            return visitors
        except Exception as e:
            logging.error(f"Error saat mengambil data pengunjung: {e}")
            return []
    
    def get_all_visitors(self):
        """
        Ambil semua data pengunjung dari database
        
        Returns:
            List semua data pengunjung [(id, timestamp, direction, notes), ...]
        """
        return self.get_visitors(after_id=0)
    
    def get_visitor_count(self):
        """
        Ambil jumlah total pengunjung
        
        Returns:
            Jumlah total pengunjung
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM visitors')
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
        except Exception as e:
            logging.error(f"Error saat mengambil jumlah pengunjung: {e}")
            return 0