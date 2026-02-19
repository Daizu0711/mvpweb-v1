from supabase import create_client, Client
import os
from datetime import datetime
import json

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')

supabase_client: Client = None


def init_supabase():
    global supabase_client
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("⚠ Supabase credentials not set")
        return None
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Supabase connected")
        return supabase_client
    except Exception as e:
        print(f"⚠ Supabase connection failed: {e}")
        return None


def get_supabase():
    global supabase_client
    if supabase_client is None:
        init_supabase()
    return supabase_client


def load_users():
    """Load all users from Supabase."""
    try:
        client = get_supabase()
        if not client:
            return {}
        
        response = client.table('users').select('*').execute()
        users = {}
        for row in response.data:
            user_id = row['username']
            users[user_id] = {
                'registered_at': row['registered_at'],
                'tpose_image': row['tpose_image'],
                'body_ratios': json.loads(row['body_ratios']) if isinstance(row['body_ratios'], str) else row['body_ratios'],
                'tpose_keypoints': json.loads(row['tpose_keypoints']) if isinstance(row['tpose_keypoints'], str) else row['tpose_keypoints']
            }
        return users
    except Exception as e:
        print(f"⚠ Failed to load users: {e}")
        return {}


def save_user(username, user_data):
    """Save or update user in Supabase."""
    try:
        client = get_supabase()
        if not client:
            return False
        
        body_ratios = json.dumps(user_data['body_ratios']) if isinstance(user_data['body_ratios'], dict) else user_data['body_ratios']
        tpose_keypoints = json.dumps(user_data['tpose_keypoints']) if isinstance(user_data['tpose_keypoints'], list) else user_data['tpose_keypoints']
        
        response = client.table('users').upsert({
            'username': username,
            'registered_at': user_data.get('registered_at', datetime.now().isoformat()),
            'tpose_image': user_data.get('tpose_image', ''),
            'body_ratios': body_ratios,
            'tpose_keypoints': tpose_keypoints
        }).execute()
        
        return True
    except Exception as e:
        print(f"⚠ Failed to save user: {e}")
        return False


def load_training_records():
    """Load training records from Supabase."""
    try:
        client = get_supabase()
        if not client:
            return []
        
        response = client.table('training_records').select('*').execute()
        records = []
        for row in response.data:
            record = {
                'timestamp': row['timestamp'],
                'compare_mode': row['compare_mode'],
                'self_improved': row['self_improved'],
                'overall_score': row['overall_score'],
                'joint_scores': json.loads(row['joint_scores']) if isinstance(row['joint_scores'], str) else row['joint_scores'],
                'temporal_alignment': row['temporal_alignment'],
                'use_3d': row['use_3d']
            }
            if row.get('metrics_3d'):
                record['metrics_3d'] = json.loads(row['metrics_3d']) if isinstance(row['metrics_3d'], str) else row['metrics_3d']
            records.append(record)
        return records
    except Exception as e:
        print(f"⚠ Failed to load training records: {e}")
        return []


def append_training_record(record):
    """Add training record to Supabase."""
    try:
        client = get_supabase()
        if not client:
            return False
        
        joint_scores = json.dumps(record['joint_scores']) if isinstance(record['joint_scores'], dict) else record['joint_scores']
        metrics_3d = json.dumps(record.get('metrics_3d')) if record.get('metrics_3d') else None
        
        client.table('training_records').insert({
            'timestamp': record['timestamp'],
            'compare_mode': record['compare_mode'],
            'self_improved': record['self_improved'],
            'overall_score': record['overall_score'],
            'joint_scores': joint_scores,
            'temporal_alignment': record['temporal_alignment'],
            'use_3d': record['use_3d'],
            'metrics_3d': metrics_3d
        }).execute()
        
        return True
    except Exception as e:
        print(f"⚠ Failed to append training record: {e}")
        return False


def load_inference_results():
    """Load inference results from Supabase."""
    try:
        client = get_supabase()
        if not client:
            return []
        
        response = client.table('inference_results').select('*').order('timestamp', desc=True).limit(100).execute()
        results = []
        for row in response.data:
            result = {
                'timestamp': row['timestamp'],
                'username': row['username'],
                'backend': row['backend'],
                'compare_mode': row['compare_mode'],
                'result': json.loads(row['result']) if isinstance(row['result'], str) else row['result']
            }
            results.append(result)
        return results
    except Exception as e:
        print(f"⚠ Failed to load inference results: {e}")
        return []


def append_inference_result(result):
    """Add inference result to Supabase."""
    try:
        client = get_supabase()
        if not client:
            return False
        
        result_json = json.dumps(result['result']) if isinstance(result['result'], dict) else result['result']
        
        client.table('inference_results').insert({
            'timestamp': result['timestamp'],
            'username': result['username'],
            'backend': result['backend'],
            'compare_mode': result['compare_mode'],
            'result': result_json
        }).execute()
        
        return True
    except Exception as e:
        print(f"⚠ Failed to append inference result: {e}")
        return False
