import yaml_parser as yp

def yaml_path(path):
    my_yaml_info=yp.YAMLOperator(path)
    yaml_dict=my_yaml_info.parse_yaml()
    return yaml_dict