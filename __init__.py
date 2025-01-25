"""
@author: Dr.Lt.Data
@title: Impact Subpack
@nickname: Impact Subpack
@description: This extension provides UltralyticsDetectorProvider node
"""

import importlib
import logging

version_code = [1, 2, 9]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
logging.info(f"### Loading: ComfyUI-Impact-Subpack ({version_str})")

node_list = [
    "subpack_nodes",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(".modules.{}".format(module_name), __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

try:
    import cm_global
    cm_global.register_extension('ComfyUI-Impact-Subpack',
                                 {'version': version_code,
                                  'name': 'Impact Subpack',
                                  'nodes': set(NODE_CLASS_MAPPINGS.keys()),
                                  'description': 'This extension provides UltralyticsDetectorProvider node.', })
except:
    pass
