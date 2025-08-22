import shutil
import os

def zip_folder(folder_path, output_path):
    """
    지정된 폴더를 ZIP 파일로 압축하는 함수.

    Args:
        folder_path (str): 압축할 폴더의 경로.
        output_path (str): 생성될 ZIP 파일의 경로 (확장자 제외).
    
    Returns:
        str: 성공 시 생성된 ZIP 파일의 전체 경로를 반환하고, 실패 시 None을 반환.
    """
    try:
        # shutil.make_archive(base_name, format, root_dir)
        # base_name: 생성될 파일 이름 (경로 포함, .zip 확장자 제외)
        # format: 압축 포맷 ('zip', 'tar', 'gztar' 등)
        # root_dir: 압축할 폴더의 경로
        archive_name = shutil.make_archive(output_path, 'zip', folder_path)
        print(f"📦 '{folder_path}' 폴더가 '{archive_name}'으로 성공적으로 압축되었습니다.")
        return archive_name
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        return None

# --- 사용 예시 ---
# 압축하고 싶은 폴더의 경로를 지정하세요.
source_folder = '/home/users/chaehyun/RS/A-LLMRec/pre_train/sasrec_multi/data' # 예시 경로입니다.
# 또는 현재 스크립트가 있는 폴더 아래의 'MyProject' 폴더를 지정할 수도 있습니다.
# source_folder = 'MyProject'

# 생성될 ZIP 파일의 경로와 이름을 지정하세요 (확장자는 제외).
# 예를 들어, 'C:\\Archives\\MyProject.zip' 파일이 생성됩니다.
output_file = '/home/users/chaehyun/RS/A-LLMRec/pre_train/sasrec_multi/data' # 예시 경로입니다.
# 또는 현재 스크립트가 있는 폴더에 'MyProject.zip' 파일을 만들 수도 있습니다.
# output_file = 'MyProject'

# 폴더 압축 함수 호출
if os.path.exists(source_folder):
    zip_folder(source_folder, output_file)
else:
    print(f"'{source_folder}' 경로의 폴더를 찾을 수 없습니다.")