from pathlib import Path
from core.utils import normalize_filename

class FileRepository:
    
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
    
    def save_file(self, file_name: str, content: bytes) -> str:
        file_path = f"{self.base_path}/{file_name}"
        # Create directory if it doesn't exist
        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as file:
            file.write(content)
        return file_path
    
    def get_file(self, file_name: str) -> bytes:
        file_path = f"{self.base_path}/{file_name}"
        with open(file_path, "rb") as file:
            return file.read()
        
    def delete_file(self, file_name: str) -> None:
        file_path = f"{self.base_path}/{file_name}"
        try:
            Path(file_path).unlink()
        except FileNotFoundError:
            pass