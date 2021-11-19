#!/usr/bin/python
# -*- coding:utf-8 -*-

import os

class Save_Tool(object):
    def __init__(self, max_num=10):
        self.save_list = []
        self.max_num = max_num

    def update(self, save_path):
        if len(self.save_list) < self.max_num:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)