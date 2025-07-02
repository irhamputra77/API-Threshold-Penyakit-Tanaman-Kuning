from flask import Flask, request, jsonify
from PIL import Image
import os
import time
import numpy as np
import cv2 as cv
import exifread
from flask_cors import CORS
import datetime

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def baca_exif_altitude(filepath):
    try:
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f)
        gps_altitude = tags.get('GPS GPSAltitude')
        gps_ref = tags.get('GPS GPSAltitudeRef')
        if gps_altitude:
            altitude = float(gps_altitude.values[0].num) / float(gps_altitude.values[0].den)
            if gps_ref and gps_ref.values[0] == 1:
                altitude = -altitude
            return f"{altitude:.2f} meters"
        else:
            return None
    except Exception:
        return None

@app.route('/api/process', methods=['POST'])
def process_image():
    if 'image' not in request.files or 'sample' not in request.files:
        return jsonify({'error': 'Both "image" and "sample" files are required.'}), 400

    image_file = request.files['image']
    sample_file = request.files['sample']

    if image_file.filename == '' or sample_file.filename == '':
        return jsonify({'error': 'Both files must be selected.'}), 400

    timestamp = str(int(time.time()))
    base_filename = f"{timestamp}"
    original_filename = f"original_{base_filename}.jpg"
    sample_filename = f"sample_{base_filename}.jpg"
    hsv_awal_filename = f"HSV-awal_{base_filename}.jpg"
    hsv_hasil_filename = f"HSV-hasil_{base_filename}.jpg"
    hasil_jpg_filename = f"hasil-JPG_{base_filename}.jpg"

    # Simpan file
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    sample_path = os.path.join(app.config['UPLOAD_FOLDER'], sample_filename)
    hsv_awal_path = os.path.join(app.config['UPLOAD_FOLDER'], hsv_awal_filename)
    hsv_hasil_path = os.path.join(app.config['UPLOAD_FOLDER'], hsv_hasil_filename)
    hasil_jpg_path = os.path.join(app.config['UPLOAD_FOLDER'], hasil_jpg_filename)

    image_file.save(original_path)
    sample_file.save(sample_path)

    # Mulai pengukuran waktu proses
    start_time = datetime.datetime.now()

    # Proses sample (ambil threshold HSV dari sample)
    gbSRGB = np.asarray(Image.open(sample_path))
    gbSHSV = cv.cvtColor(gbSRGB, cv.COLOR_BGR2HSV_FULL)

    gbRGB = np.asarray(Image.open(original_path))
    gbHSV = cv.cvtColor(gbRGB, cv.COLOR_BGR2HSV_FULL)

    # Ambil threshold sesuai kode Jupyter
    hmax = np.average(gbSHSV[:, :, 0]) + 2 * np.std(gbSHSV[:, :, 0])
    hmin = np.average(gbSHSV[:, :, 0]) - 2 * np.std(gbSHSV[:, :, 0])
    smax = np.average(gbSHSV[:, :, 1]) + 2 * np.std(gbSHSV[:, :, 1])
    smin = np.average(gbSHSV[:, :, 1]) - 2 * np.std(gbSHSV[:, :, 1])
    vmax = np.average(gbSHSV[:, :, 2]) + 5 * np.std(gbSHSV[:, :, 2])
    vmin = np.average(gbSHSV[:, :, 2]) - 5 * np.std(gbSHSV[:, :, 2])

    # Masking sesuai operator di kode Jupyter
    h = np.logical_and(gbHSV[:, :, 0] >= hmin, gbHSV[:, :, 0] <= hmax)
    s = np.logical_and(gbHSV[:, :, 1] > smin, gbHSV[:, :, 1] < smax)
    v = np.logical_and(gbHSV[:, :, 2] > vmin, gbHSV[:, :, 2] < vmax)
    mask = np.logical_and(h, np.logical_and(s, v))

    # Buat array hasil dengan tipe uint8
    gbHasil = np.zeros(gbHSV.shape, dtype=np.uint8)
    gbHasil[:, :, 0] = np.where(mask, gbHSV[:, :, 0], 0)
    gbHasil[:, :, 1] = np.where(mask, gbHSV[:, :, 1], 0)
    gbHasil[:, :, 2] = np.where(mask, gbHSV[:, :, 2], 0)

    # Convert hasil ke RGB
    hasilJPG = cv.cvtColor(gbHasil, cv.COLOR_HSV2RGB_FULL)


    # Selesai proses
    end_time = datetime.datetime.now()
    HGSVtime = (end_time - start_time).total_seconds()

    altitude = baca_exif_altitude(original_path)

    return jsonify({
        'original_image': f"/static/uploads/{original_filename}",
        'sample_image': f"/static/uploads/{sample_filename}",
        'hsv_awal_image': f"/static/uploads/{hsv_awal_filename}",
        'hsv_hasil_image': f"/static/uploads/{hsv_hasil_filename}",
        'hasil_jpg_image': f"/static/uploads/{hasil_jpg_filename}",
        'altitude': altitude,
        'processing_time_seconds': HGSVtime,
        'H_threshold': {'min': float(hmin), 'max': float(hmax)},
        'S_threshold': {'min': float(smin), 'max': float(smax)},
        'V_threshold': {'min': float(vmin), 'max': float(vmax)},
    })

if __name__ == '__main__':
    app.run()
