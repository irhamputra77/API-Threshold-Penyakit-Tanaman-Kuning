from flask import Flask, request, jsonify
from PIL import Image
import os
import time
import numpy as np
import cv2 as cv
import exifread

app = Flask(__name__)
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
    original_filename = f"original_{timestamp}.jpg"
    processed_filename = f"processed_{timestamp}.jpg"
    hsv_awal_filename = f"hsv_awal_{timestamp}.jpg"
    hsv_akhir_filename = f"hsv_akhir_{timestamp}.jpg"
    sample_filename = f"sample_{timestamp}.jpg"

    # Simpan file
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    hsv_awal_path = os.path.join(app.config['UPLOAD_FOLDER'], hsv_awal_filename)
    hsv_akhir_path = os.path.join(app.config['UPLOAD_FOLDER'], hsv_akhir_filename)
    sample_path = os.path.join(app.config['UPLOAD_FOLDER'], sample_filename)

    image_file.save(original_path)
    sample_file.save(sample_path)

    # Baca sample dan konversi ke HSV
    sample_hsv = cv.cvtColor(np.asarray(Image.open(sample_path)), cv.COLOR_BGR2HSV_FULL)
    target_hsv = cv.cvtColor(np.asarray(Image.open(original_path)), cv.COLOR_BGR2HSV_FULL)

    # Threshold dari sample
    hmax = np.average(sample_hsv[:, :, 0]) + 2 * np.std(sample_hsv[:, :, 0])
    hmin = np.average(sample_hsv[:, :, 0]) - 2 * np.std(sample_hsv[:, :, 0])
    smax = np.average(sample_hsv[:, :, 1]) + 2 * np.std(sample_hsv[:, :, 1])
    smin = np.average(sample_hsv[:, :, 1]) - 2 * np.std(sample_hsv[:, :, 1])
    vmax = np.average(sample_hsv[:, :, 2]) + 5 * np.std(sample_hsv[:, :, 2])
    vmin = np.average(sample_hsv[:, :, 2]) - 5 * np.std(sample_hsv[:, :, 2])

    # Masking
    h = np.logical_and(target_hsv[:, :, 0] >= hmin, target_hsv[:, :, 0] <= hmax)
    s = np.logical_and(target_hsv[:, :, 1] >= smin, target_hsv[:, :, 1] <= smax)
    v = np.logical_and(target_hsv[:, :, 2] >= vmin, target_hsv[:, :, 2] <= vmax)
    mask = np.logical_and(h, np.logical_and(s, v))

    hasil_hsv = np.zeros_like(target_hsv)
    hasil_hsv[:, :, 0] = np.where(mask, target_hsv[:, :, 0], 0)
    hasil_hsv[:, :, 1] = np.where(mask, target_hsv[:, :, 1], 0)
    hasil_hsv[:, :, 2] = np.where(mask, target_hsv[:, :, 2], 0)

    # Simpan hasil
    processed_rgb = cv.cvtColor(hasil_hsv, cv.COLOR_HSV2RGB_FULL)
    hsv_awal_rgb = cv.cvtColor(target_hsv, cv.COLOR_HSV2RGB_FULL)
    hsv_akhir_rgb = cv.cvtColor(hasil_hsv, cv.COLOR_HSV2RGB_FULL)

    cv.imwrite(processed_path, processed_rgb)
    cv.imwrite(hsv_awal_path, hsv_awal_rgb)
    cv.imwrite(hsv_akhir_path, hsv_akhir_rgb)

    altitude = baca_exif_altitude(original_path)

    return jsonify({
        'original_image': f"/static/uploads/{original_filename}",
        'sample_image': f"/static/uploads/{sample_filename}",
        'processed_image': f"/static/uploads/{processed_filename}",
        'hsv_awal_image': f"/static/uploads/{hsv_awal_filename}",
        'hsv_akhir_image': f"/static/uploads/{hsv_akhir_filename}",
        'altitude': altitude
    })

if __name__ == '__main__':
    app.run()
