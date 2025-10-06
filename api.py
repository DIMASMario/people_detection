from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import uvicorn
import os
import time
from datetime import datetime
import threading
import logging

app = FastAPI(title="Visitor Counting API", description="API untuk sistem penghitung pengunjung")

# Referensi ke database dan counter, akan diset dari main.py
db = None
counter = None
exporter = None

def initialize(database, visitor_counter, data_exporter):
    """Inisialisasi referensi global untuk API"""
    global db, counter, exporter
    db = database
    counter = visitor_counter
    exporter = data_exporter
    logging.info("API diinisialisasi dengan database dan counter")

@app.get("/status")
async def get_status():
    """Dapatkan status real-time dari penghitung pengunjung"""
    if not counter or not db:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    active_visitors = db.get_active_visitors()
    stats = counter.get_stats()
    
    return {
        "status": "active",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "visitor_count": stats["total"],
        "active_visitors": len(active_visitors),
        "direction_counts": {
            "total_left": stats["total_left"],
            "total_right": stats["total_right"]
        },
        "active_visitor_details": active_visitors
    }

@app.get("/history")
async def get_history(limit: int = 100, offset: int = 0):
    """Dapatkan riwayat pengunjung"""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    history = db.get_history(limit, offset)
    
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(history),
        "limit": limit,
        "offset": offset,
        "history": history
    }

@app.post("/export")
async def export_data(background_tasks: BackgroundTasks):
    """Ekspor data pengunjung ke Excel"""
    if not exporter:
        raise HTTPException(status_code=500, detail="Exporter not initialized")
    
    # Jalankan ekspor di background
    background_tasks.add_task(exporter.export_to_excel)
    
    return {
        "status": "Export started",
        "message": "Data akan diekspor ke file Excel secara background",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/download")
async def download_excel():
    """Download file Excel hasil ekspor"""
    if not exporter:
        raise HTTPException(status_code=500, detail="Exporter not initialized")
    
    file_path = exporter.excel_path
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found. Please export data first.")
    
    return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def run_api_server(port=8000):
    """Jalankan API server di thread terpisah"""
    uvicorn.run(app, host="0.0.0.0", port=port)

def start_api_server(port=8000):
    """Mulai API server di thread terpisah"""
    api_thread = threading.Thread(target=run_api_server, args=(port,), daemon=True)
    api_thread.start()
    return api_thread