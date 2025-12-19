import os
import re
import shutil


class FileManager:
    def __init__(self):
        pass

    def sniff_file(self, directory_path, pattern, max_deep=-1) -> list[str]:
        """Get all files in the directory_path that match the pattern

        Args:
            directory_path (``str``): the path of the directory to sniff
            pattern (``str``): the pattern to match
            max_deep (``int``, *optional*): the maximum depth to sniff. Defaults to -1.

        Returns:
            ``list[str]``: the full path of the files that match the pattern
        """
        ans = list()
        all_file_path_list = self.get_all_by_recursion(directory_path, max_deep=max_deep)
        for path in all_file_path_list:
            path_ = re.split(pattern="/", string=path)
            filename = path_[len(path_) - 1]
            if re.findall(pattern=pattern, string=filename):
                ans.append(path)
        return ans

    def sniff_file_by_path_pattern(self, directory_path, pattern, max_deep=-1) -> list[str]:
        """Get all files in the directory_path that match the pattern

        Args:
            directory_path (``str``): the path of the directory to sniff
            pattern (``str``): the pattern to match
            max_deep (``int``, *optional*): the maximum depth to sniff. Defaults to -1.

        Returns:
            ``list[str]``: the full path of the files that match the pattern
        """
        ans = list()
        all_file_path_list = self.get_all_by_recursion(directory_path, max_deep=max_deep)
        for path in all_file_path_list:
            if re.findall(pattern=pattern, string=path):
                ans.append(path)
        return ans

    def get_all_by_recursion(self, directory_path, max_deep):
        all_file_path_list = []

        def _get_all_by_recursion(directory_path, deep, max_deep):
            if directory_path[len(directory_path) - 1] == "/":
                directory_path = directory_path[0:-1]
            if os.path.isdir(directory_path) and (max_deep < 0 or deep < max_deep):
                filename_list = os.listdir(path=directory_path)
                for filename in filename_list:
                    _get_all_by_recursion(directory_path + "/" + filename, deep + 1, max_deep)
            else:
                all_file_path_list.append(directory_path)

        _get_all_by_recursion(directory_path, 0, max_deep)
        return all_file_path_list

    def copy_files_to_target(self, tgt_dir: str, src_file_list: list[str], micro_deep: int = 1) -> None:
        """
        Copy files in src_file_list to tgt_dir, preserving the last `micro_deep`
        levels of their directory structure.

        Example:
            src: a/b/c/d/e.txt
            micro_deep = 3
            result: tgt_dir/c/d/e.txt
        """
        if micro_deep < 1:
            raise ValueError("micro_deep must be greater than or equal to 1")

        tgt_dir = os.path.abspath(tgt_dir)

        for src_path in src_file_list:
            src_path = os.path.abspath(src_path)

            if not os.path.isfile(src_path):
                continue

            parts = src_path.split(os.sep)

            relative_parts = parts[-micro_deep:]

            relative_path = os.path.join(*relative_parts)
            dst_path = os.path.join(tgt_dir, relative_path)

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    print("start")
    fm = FileManager()
    fm.copy_files_to_target(tgt_dir="test", src_file_list=["shell_script/linter.sh"], micro_deep=2)
