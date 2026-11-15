import os
import sys
import asyncio
import httpx

# Set up environment
os.environ['SOCIETY_AI_AUTH_TOKEN'] = 'sai_b9fe129f12b901ad84efb6124057e44a'
os.environ['AGENT_NAME'] = 'agent-sabx4lpi'
os.environ['AGENT_ROUTER_API_URL'] = 'https://api.societyai.com'
os.environ['AGENT_TYPE'] = 'generic_cli'
os.environ['AGENT_COMMAND'] = 'codex exec --skip-git-repo-check {prompt}'
os.environ['SESSION_MODE'] = 'false'
os.environ['EXECUTION_MODE'] = 'standard'

# Add the core path
sys.path.insert(0, os.path.join(os.getcwd(), 'core'))

from core.config import SOCIETY_AI_AUTH_TOKEN, AGENT_NAME, AGENT_ROUTER_API_URL, AGENT_TYPE, AGENT_COMMAND, SESSION_MODE, EXECUTION_MODE

print('Config loaded:')
print(f'  AUTH_TOKEN: {SOCIETY_AI_AUTH_TOKEN[:20]}...')
print(f'  AGENT_NAME: {AGENT_NAME}')
print(f'  API_URL: {AGENT_ROUTER_API_URL}')
print(f'  AGENT_TYPE: {AGENT_TYPE}')
print(f'  SESSION_MODE: {SESSION_MODE}')
print(f'  EXECUTION_MODE: {EXECUTION_MODE}')

# Test API connection
async def test_api():
    headers = {'Authorization': f'Bearer {SOCIETY_AI_AUTH_TOKEN}', 'Content-Type': 'application/json'}
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test contacts endpoint
        resp = await client.get(f'{AGENT_ROUTER_API_URL}/v1/contacts', headers=headers)
        print(f'Contacts: {resp.status_code} - {resp.text[:200]}')
        
        # Test connector check
        resp = await client.post(f'{AGENT_ROUTER_API_URL}/v1/connector/check', headers=headers, json={'send_test': True})
        print(f'Connector check: {resp.status_code} - {resp.text[:500]}')

asyncio.run(test_api())
