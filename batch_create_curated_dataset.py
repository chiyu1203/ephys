from create_curated_dataset import main
import os
import time
def is_dir(path):
    return os.path.isdir(path.replace("\\", "/"))

def process_experiment(experiment_path, analysis_methods):
    session_list = os.listdir(experiment_path)
    if any(is_dir(os.path.join(experiment_path, session).replace("\\", "/")) for session in session_list):
        main(experiment_path, analysis_methods)

def process_database(database_path, analysis_methods):
    for dir_name in os.listdir(database_path):
        dir_path = os.path.join(database_path, dir_name).replace("\\", "/")
        if is_dir(dir_path):
            for experiment_name in os.listdir(dir_path):
                experiment_path = os.path.join(dir_path, experiment_name).replace("\\", "/")
                if is_dir(experiment_path):
                    process_experiment(experiment_path, analysis_methods)

def list_up_dataset(this_database, analysis_methods):
    #input_path='Z:/DATA/experiment_openEphys/'
    #file_starts='full'
    # vid_list=[]
    # file_type='.dat'
    # vid_list=[os.path.join(root,files).replace("\\","/")
    #     for root, dirs, files in os.walk(this_database)
    #     for name in files
    #     if name.endswith(file_type)]

##list up all files of this file type under the input path (include subdirectories)
    
    folder_list=os.listdir(this_database)
    for this_dir in folder_list:
        this_dir_path=os.path.join(this_database,this_dir).replace("\\","/")
        if os.path.isdir(this_dir_path):
            experiment_list=os.listdir(this_dir_path)
            for this_experiment in experiment_list:
                this_exp_path=os.path.join(this_dir_path,this_experiment).replace("\\","/")
                if os.path.isdir(this_exp_path):
                    session_list=os.listdir(this_exp_path)
                    if any(os.path.isdir(os.path.join(this_exp_path,this_session).replace("\\","/")) for this_session in session_list):
                        analysis_methods.update(Data_format="open_ephys")
                        main(os.path.join(this_database,this_dir).replace("\\","/"),analysis_methods)
                    else:
                        analysis_methods.update(Data_format="nwb")
                        main(os.path.join(this_database,this_dir).replace("\\","/"),analysis_methods)
                else:
                    continue
        else:
            continue




if __name__ == "__main__":
    this_database = "C:/Users/neuroLaptop/Documents/Open Ephys/"
    analysis_methods = {
        "Overwrite_curated_dataset": True,
        "Reanalyse_data": True,
        "Data_format":"open_ephys",
        "Analye_entire_recording":True,
        "Plot_trace": False,
        "Debug_mode": True,
    }
    ##Time the function
    tic = time.perf_counter()
    list_up_dataset(this_database, analysis_methods)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")