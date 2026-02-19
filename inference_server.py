from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import cv2
import traceback
from datetime import datetime

from pose_analyzer import PoseAnalyzer, PoseComparator
from deficiency import detect_deficiency, average_ratios_from_poses

app = Flask(__name__)
CORS(app)

TEMP_UPLOAD_DIR = os.environ.get('TEMP_UPLOAD_DIR', '/tmp/vireora_inference_uploads')
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

INFERENCE_USE_VITPOSE = os.environ.get('INFERENCE_USE_VITPOSE', 'true').lower() == 'true'
INFERENCE_MODEL_VARIANT = os.environ.get('INFERENCE_MODEL_VARIANT', 'vitpose-b')

analyzer = None


def init_analyzer(use_vitpose=True, model_variant='vitpose-b'):
    global analyzer
    if analyzer is None:
        analyzer = PoseAnalyzer(use_vitpose=use_vitpose, model_variant=model_variant)
    return analyzer


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'ok',
        'service': 'vireora-colab-inference',
        'timestamp': datetime.now().isoformat(),
        'use_vitpose': INFERENCE_USE_VITPOSE,
        'model_variant': INFERENCE_MODEL_VARIANT
    })


@app.route('/infer', methods=['POST'])
def infer():
    ref_path = None
    comp_path = None

    try:
        if 'reference_video' not in request.files or 'comparison_video' not in request.files:
            return jsonify({'success': False, 'error': 'reference_video and comparison_video are required'}), 400

        reference_file = request.files['reference_video']
        comparison_file = request.files['comparison_video']

        use_3d = request.form.get('use_3d', 'false').lower() == 'true'
        use_vitpose = request.form.get('use_vitpose', str(INFERENCE_USE_VITPOSE).lower()).lower() == 'true'
        model_variant = request.form.get('model_variant', INFERENCE_MODEL_VARIANT)

        username = request.form.get('username', '').strip()
        registered_ratios_raw = request.form.get('registered_ratios', '')
        registered_ratios = None
        if registered_ratios_raw:
            try:
                registered_ratios = json.loads(registered_ratios_raw)
            except Exception:
                registered_ratios = None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        ref_path = os.path.join(TEMP_UPLOAD_DIR, f'reference_{timestamp}.mp4')
        comp_path = os.path.join(TEMP_UPLOAD_DIR, f'comparison_{timestamp}.mp4')

        reference_file.save(ref_path)
        comparison_file.save(comp_path)

        pose_analyzer = init_analyzer(use_vitpose=use_vitpose, model_variant=model_variant)

        reference_poses = pose_analyzer.extract_poses_from_video(ref_path)
        comparison_poses = pose_analyzer.extract_poses_from_video(comp_path)

        if not reference_poses or not comparison_poses:
            return jsonify({'success': False, 'error': 'Failed to extract poses from videos'}), 500

        comparator = PoseComparator()
        comparison_result = comparator.compare_pose_sequences(
            reference_poses,
            comparison_poses,
            use_3d=use_3d
        )

        deficiencies = []
        if registered_ratios:
            avg_ratios = average_ratios_from_poses(comparison_poses)
            if avg_ratios:
                deficiencies = detect_deficiency(avg_ratios, registered_ratios)

        return jsonify({
            'success': True,
            'score': comparison_result['overall_score'],
            'joint_scores': comparison_result['joint_scores'],
            'temporal_alignment': comparison_result['temporal_alignment'],
            'frame_scores': comparison_result.get('frame_scores', []),
            'use_3d': use_3d,
            'username': username,
            'deficiencies': deficiencies,
            'analysis': '',
            'inference_backend': 'colab'
        })

    except Exception as e:
        print(f'Inference server error: {e}')
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

    finally:
        for file_path in [ref_path, comp_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass


if __name__ == '__main__':
    print('Starting Colab inference server...')
    init_analyzer()
    app.run(host='0.0.0.0', port=8000, debug=False)
