#####RUN ONLY ONCE THE FIRST TIME YOU LOAD THE PROJECT 
#######AND THEN RESTART



print("running script")
import bpy
import os
from pathlib import Path

def ShowMessageBox(message = "", title = "Message Box", icon = 'INFO'):

    def draw(self, context):
        self.layout.label(message)

    bpy.context.window_manager.popup_menu(draw, title = title, icon = icon)


class MessageBoxOperator(bpy.types.Operator):
    bl_idname = "ui.show_message_box"
    bl_label = "Minimal Operator"

    def execute(self, context):
        #this is where I send the message
        self.report({'INFO'}, "MUST RELOAD BLENDER ONCE TO CONFIG PYTHON!!!!")
        return {'FINISHED'}




def set_Up_Modules_and_key_config():
    

    dir_path = os.path.dirname(os.path.realpath(__file__))

    data_folder = Path(dir_path)
    just_Folder = data_folder.parents[0]
    
    bpy.ops.wm.keyconfig_import(filepath=str(just_Folder / "keyboard_config.py"))
    
    complete_path = just_Folder / "blender_scripts"
    print(complete_path)
    bpy.context.user_preferences.filepaths.script_directory = str(complete_path)

    bpy.ops.wm.save_userpref()
    
    #Shows a message box with a specific message 
    ShowMessageBox("Must Exit and Reload Blender to Config Python") 
    
    bpy.utils.register_class(MessageBoxOperator)

    # test call to the 
    bpy.ops.ui.show_message_box()



print("Printing script directory")
print(bpy.context.user_preferences.filepaths.script_directory)


import os
currentDir = os.getcwd()


if len(bpy.context.user_preferences.filepaths.script_directory) < 4 : 
    print("Python Script Path DOESN'T exists")
    set_Up_Modules_and_key_config()
        
else:
    print("Python Script Path already exists-but need to make sure for right system")
    
    if "/" in bpy.context.user_preferences.filepaths.script_directory:
        Script_Mac_Flag = 1
        print("Script_Mac_Flag = " + str(Script_Mac_Flag))
    else:
        Script_Mac_Flag = 0
        print("Script_Mac_Flag = " + str(Script_Mac_Flag))
    
    currentDir = os.getcwd()
    if "/" in currentDir:
        Computer_Mac_Flag = 1
        print("Computer_Mac_Flag = " + str(Computer_Mac_Flag))
    else:
        Computer_Mac_Flag = 0
        print("Computer_Mac_Flag = " + str(Computer_Mac_Flag))
    
    #if they are not of the same type then run the setup script
    if Computer_Mac_Flag != Script_Mac_Flag:
        print("script was configured for different type of system")
        #make it run the setup Script
        set_Up_Modules_and_key_config()

    else:
        #make sure that the current path is still the same
        currentPath = bpy.context.user_preferences.filepaths.script_directory
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_folder = Path(dir_path)
        just_Folder = data_folder.parents[0]
        complete_path = just_Folder / "blender_scripts"
        
        if str(complete_path) == currentPath:
            print("modules already set up")
            import datajoint
        else:
            print("current_Path does not match the saved path")
            set_Up_Modules_and_key_config()
            
    