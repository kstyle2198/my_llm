import os

def delete_files_with_extension(folder_path, extension):
    # 폴더 내의 모든 파일과 폴더 목록을 가져옵니다.
    file_list = os.listdir(folder_path)

    # 폴더 내의 각 파일에 대해 작업합니다.
    for file_name in file_list:
        # 파일의 전체 경로를 구성합니다.
        file_path = os.path.join(folder_path, file_name)

        # 파일인지 확인하고 지정된 확장자를 가진 파일이면 삭제합니다.
        if os.path.isfile(file_path) and file_name.endswith(extension):
            os.remove(file_path)
            print(f"{file_name}이(가) 삭제되었습니다.")

# 특정 폴더와 확장자를 지정하여 함수를 호출합니다.

folder_path = "C:\my_develop2\my_llm"
extensions = [".docx", ".pptx", ".pdf", "PDF", 'png', 'csv']
for ext in extensions:
    delete_files_with_extension(folder_path, ext)