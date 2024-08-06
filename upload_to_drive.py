# !pip install google-api-python-client
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload
import os
import argparse as arg


def authenticate():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = './service_account.json'
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return creds

def get_file_id_by_name(file_name, service, folder_id):
    query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get('files', [])
    return files[0]['id'] if files else None

def upload_file(local_file_path, dest_file_name):
    PARENT_FOLDER_ID = '1cl11_T-lL9rBxwclgdRtXYBX6ciHZpH7'
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    # Check if the file already exists in Google Drive
    existing_file_id = get_file_id_by_name(dest_file_name, service, PARENT_FOLDER_ID)
    if existing_file_id:
        # If the file exists, update its content
        media = MediaFileUpload(local_file_path, resumable=True)
        file = service.files().update(
            fileId=existing_file_id,
            media_body=media,
        ).execute()
        print('File updated. File ID: ', existing_file_id)
        return existing_file_id
    else:
        # If the file doesn't exist, create a new file
        file_metadata = {
            'name': dest_file_name,
            'parents': [PARENT_FOLDER_ID]
        }
        media = MediaFileUpload(local_file_path, resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
        ).execute()
        file_id = file['id']
        print('File uploaded. File ID: ', file_id)
        return file_id
    
parser = arg.ArgumentParser(description="Params")
parser.add_argument("--folder_name", type=str)
parser.add_argument("--folder_id", type=str)
args = parser.parse_args()
args = vars(args)
folder_name = args['folder_name']
folder_id = args['folder_id']
file_list = os.listdir(folder_name)

for file in file_list:
    path = folder_name + '/' + file
    print('path_file: ', path)
    upload_file(path, path)
    
# python upload_file_to_drive.py --folder_name "outputs" --folder_id "1cl11_T-lL9rBxwclgdRtXYBX6ciHZpH7"