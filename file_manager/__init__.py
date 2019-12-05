import os
from functools import reduce


class DirManager:
    def __init__(self, dir_path):
        """path，all_dirs，all_filename"""
        try:
            self.path, self.dir_list, self.file_name_list = list(os.walk(dir_path))[0]
        except Exception as e:
            raise "path error" + ":" + str(e)

    def get_all_file_name(self):
        """return all file abspath (all abs path in this path including subdirs)"""
        return reduce(lambda x, y: x + y, [[item[2][i] for i in range(len(item[2]))] for item in
                                           os.walk(
                                               self.path)])

    def get_file_abspath(self):
        """"""
        return_list = []
        for item in os.walk(self.path):
            return_list += [os.path.join(item[0], _) for _ in item[-1]]
        return return_list

    def file_filter(self, post_fix_set):
        """
        :param post_fix_set:{"jpg", "png", "json", "xml"}
        :return: [[], ['d:file_utils.xml'], ['d:.idea\\encodings.xml', 'd:.idea\\file_manager.png', 'd:.idea\\misc.xml']
        """
        files = self.get_file_abspath()
        return list(filter(lambda x: x.split(".")[-1] in post_fix_set, files))


if __name__ == '__main__':
    a = DirManager("D:\\training_data")
    print(a.file_filter({"pts"}))
