def main(thisDir, analysis_methods):

if __name__ == "__main__":
    thisDir = "C:/Users/neuroLaptop/Documents/Open Ephys/"
    analysis_methods = {
        "Overwrite_curated_dataset": True,
        "Reanalyse_data": True,
        "Fig_dir":"Z:/DATA/experiment_openEphys/GN00001",
        "Analye_entire_recording":True,
        "Plot_trace": False,
        "Debug_mode": True,
    }
    ##Time the function
    tic = time.perf_counter()
    main(thisDir, analysis_methods)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")