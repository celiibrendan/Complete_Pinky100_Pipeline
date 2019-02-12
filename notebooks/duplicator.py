def make_copies(file_name,output_file,number_files=50):
    #generate the copies
    from shutil import copyfile

    total_file_lists = list()
    total_file_lists.append(file_name)
    for i in range(0,number_files):
        #create new name for file
        new_name = file_name[:-3] +"_"+ str(i) + ".py"
        #copy the file
        copyfile(file_name, new_name)
        #add to the total list
        total_file_lists.append(new_name)


    #create theh bash file for the list
    filename = output_file
    f = open(filename, "w")
    f.write("#!/bin/bash\n")
    for ll in total_file_lists:
        f.write("python " + ll + " &\n")
    f.close()
   