#coding=utf-8
import json
import os, sys
import argparse
import datetime

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(dictionary)
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    return config_dict

def process_config(json_file):
    config = get_config_from_json(json_file)
    return config
