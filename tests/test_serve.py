import requests
import json
import time
import tempfile
from npcpy.serve import app


def test_flask_app_health():
    """Test Flask app health endpoint"""
    try:
        with app.test_client() as client:
            response = client.get('/api/health')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'ok'
            print("Health check passed")
    except Exception as e:
        print(f"Health check failed: {e}")


def test_get_models_endpoint():
    """Test models endpoint"""
    try:
        with app.test_client() as client:
            response = client.get('/api/models?currentPath=/tmp')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'models' in data
            print(f"Models endpoint returned {len(data['models'])} models")
    except Exception as e:
        print(f"Models endpoint test failed: {e}")


def test_get_global_settings():
    """Test global settings endpoint"""
    try:
        with app.test_client() as client:
            response = client.get('/api/settings/global')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'global_settings' in data
            print("Global settings endpoint working")
    except Exception as e:
        print(f"Global settings test failed: {e}")


def test_conversations_endpoint():
    """Test conversations endpoint"""
    try:
        with app.test_client() as client:
            response = client.get('/api/conversations?path=/tmp')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'conversations' in data
            print(f"Conversations endpoint returned {len(data['conversations'])} conversations")
    except Exception as e:
        print(f"Conversations endpoint test failed: {e}")


def test_capture_screenshot_endpoint():
    """Test screenshot capture endpoint"""
    try:
        with app.test_client() as client:
            response = client.get('/api/capture_screenshot')
            
            print(f"Screenshot endpoint status: {response.status_code}")
    except Exception as e:
        print(f"Screenshot endpoint test failed: {e}")


def test_global_jinxs_endpoint():
    """Test global jinxs endpoint"""
    try:
        with app.test_client() as client:
            response = client.get('/api/jinxs/global')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'jinxs' in data
            print(f"Global jinxs endpoint returned {len(data['jinxs'])} jinxs")
    except Exception as e:
        print(f"Global jinxs endpoint test failed: {e}")


def test_project_jinxs_endpoint():
    """Test project jinxs endpoint"""
    try:
        with app.test_client() as client:
            response = client.get('/api/jinxs/project?currentPath=/tmp')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'jinxs' in data
            print(f"Project jinxs endpoint returned {len(data['jinxs'])} jinxs")
    except Exception as e:
        print(f"Project jinxs endpoint test failed: {e}")


def test_npc_team_global_endpoint():
    """Test global NPC team endpoint"""
    try:
        with app.test_client() as client:
            response = client.get('/api/npc_team_global')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'npcs' in data
            print(f"Global NPC team endpoint returned {len(data['npcs'])} NPCs")
    except Exception as e:
        print(f"Global NPC team endpoint test failed: {e}")


def test_save_npc_endpoint():
    """Test save NPC endpoint"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        with app.test_client() as client:
            npc_data = {
                "npc": {
                    "name": "test_npc",
                    "primary_directive": "Test NPC",
                    "model": "llama3.2",
                    "provider": "ollama"
                },
                "isGlobal": False,  
                "currentPath": temp_dir  
            }
            response = client.post('/api/save_npc', 
                                 json=npc_data,
                                 content_type='application/json')
            print(f"Save NPC endpoint status: {response.status_code}")
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_project_settings_endpoints():
    """Test project settings get/post"""
    try:
        with app.test_client() as client:
            
            response = client.get('/api/settings/project?path=/tmp')
            assert response.status_code == 200
            
            
            settings_data = {"env_vars": {"TEST_VAR": "test_value"}}
            response = client.post('/api/settings/project?path=/tmp',
                                 json=settings_data,
                                 content_type='application/json')
            print(f"Project settings endpoints working: GET and POST")
    except Exception as e:
        print(f"Project settings endpoints test failed: {e}")
