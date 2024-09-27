import collections

def get_config_section(config):
    section_dict = collections.defaultdict()
    
    for section in config.sections():
        section_dict[section] = dict(config.items(section))
    return section_dict