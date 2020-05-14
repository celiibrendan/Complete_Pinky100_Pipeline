from pathlib import Path
from io import StringIO
import subprocess
import trimesh
import time
import copy
import os
import random


class Scripter:
    def __init__(self, filters=None, deepcopy=True):
        """
        filters is a dict, generally generated by this class
        """
        
        self._deepcopy = deepcopy
        
        if filters is None:
            self._filters = {}
        else:
            self._filters = self._copy(filters)
            
        self.test_filters()
    
    def test_filters(self):
        filters = self._filters
        try:
            for filter_name, data in filters.items():
                for param_name, param in data.items():
                    pass
        except:
            raise TypeError('filters not in correct preliminary format!')
    
    def _copy(self, _dict):
        if self._deepcopy:
            _dict = copy.deepcopy(_dict)
        else:
            _dict = _dict.copy()
        return _dict
    
    @staticmethod
    def create_param(name, type, value):
        return {
            name: {
                'type': str(type),
                'value': str(value)
            }
        }
    
    @staticmethod
    def create_filter(name, params=None, deepcopy=True):        
        if params is None:
            params = {}
        
        if deepcopy:
            params = copy.deepcopy(params)
        else:
            params = params.copy()
        
        if isinstance(params, list):
            params = self.combine_dicts(params)
        
        return {
            name: {
                **params
            }
        }
    
    @staticmethod
    def combine_dicts(list_of_dicts):
        _dict = {}
        
        for sub_dict in list_of_dicts:
            _dict.update(sub_dict)
            
        return _dict
    
    def add_params(self, filter_name, params):
        params = self._copy(params)
        
        if isinstance(params, list):
            params = self.combine_dicts(params)
        
        self._filters[filter_name].update(params)
    
    def add_filters(self, filters):
        """
        params can either be a proper dictionary or a list of dictionaries
        """
        
        filters = self._copy(filters)
                
        if isinstance(filters, list):
            filters = self.combine_dicts(filters)
            
        self._filters.update(filters)
    
    def set_param(self, filter_name, param_name, param_type, param_value):
        self._filters[filter_name][param_name] = self.create_param(param_name, param_type, param_value)
    
    def set_filter(self, filter_name, params):
        params = self._copy(params)
        
        if isinstance(params, list):
            params = self.combine_dicts(params)
        
        self._filters[filter_name] = params
    
    def adjust_param(self, filter_name, param_name, new_value):
        self._filters[filter_name][param_name]['value'] = str(new_value)
    
    def delete_param(self, filter_name, param_name):
        del self._filters[filter_name][param_name]
        
    def delete_filter(self, filter_name):
        del self._filters[filter_name]
    
    #defining the getter
    @property 
    def filters(self):
        return self._filters
    
    #defining the getter for the string buffer to write to file based on the filters
    @property
    def mls_buffer(self):
        buffer = ''
        with StringIO() as f:
            f.writelines('\n'.join(['<!DOCTYPE FilterScript>', '<FilterScript>\n\n']))

            for filter_name, data in self.filters.items():
                filter_type = "filter"
                param_type = "Param"
                if len(data) > 0:
                    if "type" not in data[list(data.keys())[0]].keys():
                        filter_type = "xmlfilter"
                        param_type = "xmlparam"
                print(f"filter_type = {filter_type}")
                
                f.write('<{filter_type} name="{name}"{spacer}>\n'.format(
                    filter_type=filter_type,
                    name=filter_name,
                    spacer=(' ' if len(data) > 0 else '/')
                ))

                for param_name, param in data.items():
                    if filter_type == "filter":
                        f.write('    <{param_type} type="{type}" value="{value}" name="{name}" />\n'.format(
                            param_type=param_type,
                            type=param['type'],
                            value=param['value'],
                            name=param_name
                        ))
                    else:
                        f.write('    <{param_type} value="{value}" name="{name}" />\n'.format(
                            param_type=param_type,
                            value=param['value'],
                            name=param_name
                        ))

                if len(data) > 0:
                    f.write(f'</{filter_type}>\n')
                f.write('\n')

            f.write('</FilterScript>\n')
            buffer = f.getvalue()
        return buffer
    
    def to_file(self, filepath, overwrite=False, error=False):
        file_obj = Path(filepath).absolute()
        if not overwrite and file_obj.exists():
            if error:
                raise FileExistsError('file at path {} already exists, func not set to overwrite'.format(filepath))
            else:
                print('filepath {} exists, skipping'.format(filepath))
        with open(file_obj, 'w') as f:
            f.write(self.mls_buffer)
            
    def __str__(self):
        return self.mls_buffer


class Meshlab:
    def __init__(self, mls_script_path):
        self.mls_script_obj = Path(mls_script_path).absolute()
        if not self.mls_script_obj.exists():
            raise FileNotFoundError('meshlab script missing')
        if self.mls_script_obj.suffix != '.mls':
            raise TypeError('meshlab script path must be to an .mls file')
            
    @staticmethod
    def mesh_to_off(vertices, faces, output_path=None):
        if output_path is None:
            output_path = Path().absolute()
        else:
            output_path = Path(output_path).absolute()
            
        trimesh.Trimesh(vertices=vertices, faces=faces).export(output_path)        
        return output_path
        
    @staticmethod
    def fetch_mesh_from_off(mesh_path):        
        mesh_obj = Path(mesh_path).absolute()
        
        if not mesh_obj.exists():
            raise FileNotFoundError('Mesh file missing')
        if mesh_obj.suffix != '.off':
            raise TypeError('mesh path must be to an .off file')
        
        return trimesh.load_mesh(mesh_obj)
        
    def __call__(self, input_mesh_path, output_mesh_path='.', printout=True):
        input_obj = Path(input_mesh_path).absolute()
        
        if not input_obj.exists():
            raise FileNotFoundError('Input file missing')
        if input_obj.suffix != '.off':
            raise TypeError('input path must be to an .off file')
        
        output_obj = Path(output_mesh_path).absolute()

        script_command = (" -i " + str(input_obj) + " -o " + 
                                        str(output_obj) + " -s " + str(self.mls_script_obj))

        command_to_run = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@ ' + script_command
        from subprocess import PIPE
        if printout:
            print(command_to_run)
        subprocess_result = subprocess.run(command_to_run, shell=True,stdout=PIPE, stderr=PIPE)
        if subprocess_result.returncode != 0:
            print(f"\n---- meshlab output -----\n"
                 f"{subprocess_result.stdout.decode()}"
                  f"\n\n returncode ====== {subprocess_result.returncode}"
                  f"\n\n ------ Done with meshlab output------")

        return subprocess_result
    
    
class Decimator(Meshlab):
    """
    Use like so:
        ```python
        # Preprocessing
        temp_folder = 'decimation_temp'
        mls_func = Decimator(0.05, temp_folder, overwrite=True)
        
        # Processing
        decimated_mesh = mls_func(mesh.vertices, mesh.faces, mesh.segment_id)        
        
        
        
    # Alternative way so don't have to read in and out files
    
    temp_folder = "new_temp"
    my_Dec = Decimator(0.35,temp_folder,overwrite=True)
    #how to run an actual decimation
    input_mesh_path = "./107738877133006848/107738877133006848_soma_3.off"

    new_mesh_decimated = my_Dec(input_mesh_path=input_mesh_path,
                               delete_temp_files=False,
                               return_mesh=False)
    """
    
    def __init__(self, decimation_ratio, temp_folder, overwrite=False, **kwargs):
        mls_script_path, folder_obj = self.preprocessing(
            decimation_ratio=decimation_ratio,
            temp_folder=temp_folder,
            overwrite=overwrite,
            **kwargs
        )
        self.temp_folder_obj = Path(temp_folder).absolute()
        super().__init__(mls_script_path) #sets the path of the mls script
    
    @staticmethod
    def initialize_script_filters(decimation_ratio):
        default_filters = {
            'Remove Duplicate Vertices': {},
            'Simplification: Quadric Edge Collapse Decimation': {
                'TargetFaceNum': dict(type='RichInt', value='100000'),
                'TargetPerc': dict(type='RichFloat', value=str(decimation_ratio)),
                'QualityThr': dict(type='RichFloat', value='1'),
                'PreserveBoundary': dict(type='RichBool', value='true'),
                'BoundaryWeight': dict(type='RichFloat', value='1'),
                'PreserveNormal': dict(type='RichBool', value='true'),
                'PreserveTopology': dict(type='RichBool', value='true'),
                'OptimalPlacement': dict(type='RichBool', value='true'),
                'PlanarQuadric': dict(type='RichBool', value='true'),
                'PlanarWeight': dict(type='RichFloat', value='1'),
                'QualityWeight': dict(type='RichBool', value='false'),
                'AutoClean': dict(type='RichBool', value='true'),
                'Selected': dict(type='RichBool', value='false')
            }
        }
        return default_filters
    
    @classmethod
    def create_decimation_script(cls,folder_path, decimation_ratio, output_path=None, overwrite=False, **custom_filters):
        """
        Actually writes the decimation script to file
        """
        
        if output_path is None:
            #output_path_folder = Path() / "Decimation_scripts"
            output_path_folder = Path(folder_path)
            if not output_path_folder.exists():
                output_path_folder.mkdir()
            output_path = output_path_folder / 'decimation_meshlab_{:02}{}.mls'.format(int(decimation_ratio*100),random.randint(0,999999))
        
        dec_filter_name = 'Simplification: Quadric Edge Collapse Decimation'
        dec_param_name = 'TargetPerc'
        
         #sending the filters to the Scripter
        decimation_mls = Scripter(cls.initialize_script_filters(decimation_ratio))
        
        #added extra custom filters
        if len(custom_filters) > 0:
            decimation_mls.add_filters(custom_filters) 
        # decimation_mls.adjust_param(dec_filter_name, dec_param_name, decimation_ratio) # not needed with the initializer
        
        decimation_mls.to_file(output_path, overwrite=overwrite)
        
        return output_path
    
    @classmethod
    def preprocessing(cls, decimation_ratio, temp_folder='.', **kwargs):
        folder_obj = Path(temp_folder)
        
        #makes the folder if doesn't already exist
        if not folder_obj.exists():
            folder_obj.mkdir()

        #creates the decimation script
        mls_script_path = cls.create_decimation_script(str(folder_obj.absolute()),decimation_ratio, **kwargs)

        return mls_script_path, folder_obj
    
    def __call__(self, vertices=[], faces=[], segment_id=None, 
                 return_mesh= True,
                 input_mesh_path="",
                 mesh_filename="",
                 printout=True, delete_temp_files=True):
        
        if len(input_mesh_path) <= 0:
            if len(mesh_filename)<=0:
                mesh_filename = 'neuron_{}.off'.format(segment_id)
            input_mesh_path = self.mesh_to_off(vertices, faces, output_path=(self.temp_folder_obj / mesh_filename))
        
        input_obj = Path(input_mesh_path).absolute()
        if not input_obj.exists():
            raise FileNotFoundError('input file for poission not found') 
        
        try_counter = 10
        for i in range(try_counter):
            print('IN INPUT FILE VALIDATION LOOP')
            try:
                input_mesh = self.fetch_mesh_from_off(input_mesh_path)
                print('LEAVING LOOP, MESH VALIDATED')
                break
            except Exception as e:
                print('VALIDATION ERROR (sleepin): ' + str(e))
                time.sleep(2)
            
            if (i + 1) >= try_counter:
                raise ValueError('MESH VALIDATION TRIES EXCEEDED')
        
        output_obj = self.temp_folder_obj / '{}_decimated{}'.format(input_obj.stem, input_obj.suffix)
        
        subprocess_result = super().__call__(
            input_mesh_path=input_obj,
            output_mesh_path=output_obj,
            printout=printout
        )
        
        if subprocess_result.returncode != 0:
            raise Exception('neuron {} did not fix the manifold edges (meshlab script failed) with output:{} '.format(segment_id,subprocess_result.stdout.decode())) 
#         if str(subprocess_result)[-13:] != "returncode=0)":
#             raise Exception('neuron {} did not fix the manifold edges (meshlab script failed)'.format(segment_id))
        
        if return_mesh:
            current_mesh = self.fetch_mesh_from_off(str(output_obj))
        
        
        if delete_temp_files:
            if input_obj.is_file():
                os.remove(input_obj)
                print('removed temporary input file: {}'.format(input_obj))
            if output_obj.is_file():
                os.remove(output_obj)
                print('removed temporary output file: {}'.format(output_obj))

        if return_mesh:
            return current_mesh,output_obj
        else:
            return output_obj
    def __enter__(self): 
        return self
  
    def __exit__(self,exception_type, exception_value, traceback): 
        #delete the Poisson file
        print(f"{str(self.mls_script_obj)} is being deleted....")
        if self.mls_script_obj.exists():
            self.mls_script_obj.unlink()
    
    
class Poisson(Meshlab):
    """
    Use like so:
        ```python
        # Preprocessing
        temp_folder = 'decimation_temp'
        mls_func = Decimator(0.05, temp_folder, overwrite=True)
        
        # Processing
        decimated_mesh = mls_func(mesh.vertices, mesh.faces, mesh.segment_id)        
        
        
        #Alternative way where don't have to read in and out files
        temp_folder = "poisson_temp"
        my_Poisson = Poisson(temp_folder,overwrite=True)
        #how to run an actual decimation
        input_mesh_path = "./107738877133006848/107738877133006848_soma_3.off"
        new_mesh_decimated = my_Poisson(input_mesh_path=input_mesh_path,
                                       delete_temp_files=False,
                                       return_mesh=True)
        
    """
    
    def __init__(self, temp_folder, overwrite=False, **kwargs):
        mls_script_path, folder_obj = self.preprocessing(
            temp_folder=temp_folder,
            overwrite=overwrite,
            **kwargs
        )
        self.temp_folder_obj = Path(temp_folder).absolute()
        super().__init__(mls_script_path) #sets the path of the mls script
    
    @staticmethod
    def initialize_script_filters():
        default_filters = {
            'Remove Duplicate Vertices': {},
            'Smooths normals on a point sets': {
                'K': dict(type='RichInt', value='10'),
                'useDist': dict(type='RichBool', value='false'),
            },
            'Surface Reconstruction: Screened Poisson': {
                'cgDepth': dict(value='0'),
                'confidence': dict(value='false'),
                'depth': dict(value='11'),
                'fullDepth': dict(value='6'),
                'iters': dict(value='8'),
                'pointWeight': dict(value='4'),
                'preClean': dict(value='false'),
                'samplesPerNode': dict(value='1.5'),
                'scale': dict(value='1.1'),
                'visibleLayer': dict(value='false'), 
            },
            'Delete Current Mesh': {},
        }
        return default_filters
    
    @classmethod
    def create_poisson_script(cls,folder_path,output_path=None, overwrite=False, **custom_filters):
        """
        Actually writes the decimation script to file
        """
        if output_path is None:
            output_path_folder = Path(folder_path)
            if not output_path_folder.exists():
                output_path_folder.mkdir()
            output_path = output_path_folder / f'poisson_{random.randint(0,999999)}.mls'
        
         #sending the filters to the Scripter
        poisson_mls = Scripter(cls.initialize_script_filters())
        
        #added extra custom filters
        if len(custom_filters) > 0:
            poisson_mls.add_filters(custom_filters) 

        poisson_mls.to_file(output_path, overwrite=overwrite)
        
        return output_path
    
    @classmethod
    def preprocessing(cls, temp_folder='.', **kwargs):
        folder_obj = Path(temp_folder)
        
        #makes the folder if doesn't already exist
        if not folder_obj.exists():
            folder_obj.mkdir()

        #creates the decimation script
        mls_script_path = cls.create_poisson_script(str(folder_obj.absolute()),**kwargs)

        return mls_script_path, folder_obj
    
    def __call__(self, vertices=[], faces=[], segment_id=None,
                 return_mesh= True,
                 input_mesh_path="",
                 mesh_filename="",
                 printout=True, delete_temp_files=True):
        
        if len(input_mesh_path) <= 0:
            if len(mesh_filename)<=0:
                mesh_filename = 'neuron_{}.off'.format(segment_id)
            input_mesh_path = self.mesh_to_off(vertices, faces, output_path=(self.temp_folder_obj / mesh_filename))
        
        input_obj = Path(input_mesh_path).absolute()
        if not input_obj.exists():
            raise FileNotFoundError('input file for poission not found') 
        
        try_counter = 10
        for i in range(try_counter):
            print('IN INPUT FILE VALIDATION LOOP')
            try:
                input_mesh = self.fetch_mesh_from_off(input_mesh_path)
                print('LEAVING LOOP, MESH VALIDATED')
                break
            except Exception as e:
                print('VALIDATION ERROR (sleepin): ' + str(e))
                time.sleep(2)
            
            if (i + 1) >= try_counter:
                raise ValueError('MESH VALIDATION TRIES EXCEEDED')
        
        output_obj = self.temp_folder_obj / '{}_poisson{}'.format(input_obj.stem, input_obj.suffix)
        
        subprocess_result = super().__call__(
            input_mesh_path=input_obj,
            output_mesh_path=output_obj,
            printout=printout
        )
        
        if subprocess_result.returncode != 0:
            raise Exception('neuron {} did not fix the manifold edges (meshlab script failed) with output:{} '.format(segment_id,subprocess_result.stdout.decode())) 
#         if str(subprocess_result)[-13:] != "returncode=0)":
#             raise Exception('neuron {} did not fix the manifold edges (meshlab script failed)'.format(segment_id))
        if return_mesh:
            current_mesh = self.fetch_mesh_from_off(str(output_obj))
        
        if delete_temp_files:
            if input_obj.is_file():
                os.remove(input_obj)
                print('removed temporary input file: {}'.format(input_obj))
            if output_obj.is_file():
                os.remove(output_obj)
                print('removed temporary output file: {}'.format(output_obj))
        
        if return_mesh:
            return current_mesh,output_obj
        else:
            return output_obj
    def __enter__(self): 
        return self
  
    def __exit__(self,exception_type, exception_value, traceback): 
        #delete the Poisson file
        print(f"{str(self.mls_script_obj)} is being deleted....")
        if self.mls_script_obj.exists():
            self.mls_script_obj.unlink()
        