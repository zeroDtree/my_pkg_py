import os
import re


class Sniffer:
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
        all_file_path_list = self.get_all_by_recursion(directory_path, deep=0, max_deep=max_deep)
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
        all_file_path_list = self.get_all_by_recursion(directory_path, deep=0, max_deep=max_deep)
        for path in all_file_path_list:
            if re.findall(pattern=pattern, string=path):
                ans.append(path)
        return ans

    def get_all_by_recursion(self, directory_path, deep, max_deep):
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

        _get_all_by_recursion(directory_path, deep, max_deep)
        return all_file_path_list


if __name__ == "__main__":
    sniffer = Sniffer()
    print(sniffer.sniff_file("text", "", 1))
