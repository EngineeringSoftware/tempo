import sys
import pandas as pd
from io import StringIO

WARP_EXECUTION_EFFICIENCY: str = "smsp__thread_inst_executed_per_inst_executed.ratio"
ACHIEVED_OCCUPANCY: str = "sm__warps_active.avg.pct_of_peak_sustained_active"
GLOBAL_STORE_THROUGHPUT: str = "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second"
GLOBAL_LOAD_THROUGHPUT: str = "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second"

def printStatistics(size, name, column):
    if explore_type == "exactthreads":
        number = "," + str(results.shape[0])
    else:
        number = ""
    if name == "GST" or name == "GLT":
        column = column.div(10 ** 9)
    print(size + "," + name +
        "," + "{0:.2f}".format(column.mean()) +
        "," + "{0:.2f}".format(column.min()) +
        "," + "{0:.2f}".format(column.max()) +
        "," + "{0:.2f}".format(column.median()) +
        "," + "{0:.2f}".format(column.std()) +
        number)



if len(sys.argv) != 4:
    sys.exit("ERROR: python3 " + sys.argv[0] + " input_file size explore_type")

input_file: str = sys.argv[1]
size: str = sys.argv[2]
explore_type: str = sys.argv[3]

with open(input_file) as f:
    results = f.readlines()
    index = -1
    for idx, line in enumerate(results):
        if "==PROF== Disconnected" in line:
            index = idx

    if index == -1:
        sys.exit(input_file + " has no profiling data")

    results = results[index + 1:]
    results = "".join(results)
    results = StringIO(results)
    results = pd.read_csv(results)
    results.drop(results.loc[results["Kernel Name"] == "_copyWorklistKernel(int*,int*,unsigned int)"].index, inplace=True)
    if explore_type == "rangethreads":
        results = results.iloc[1:]

    results[ACHIEVED_OCCUPANCY] = pd.to_numeric(results[ACHIEVED_OCCUPANCY], errors = "coerce")

    results[GLOBAL_STORE_THROUGHPUT] = results[GLOBAL_STORE_THROUGHPUT].str.replace(",", "")
    results[GLOBAL_STORE_THROUGHPUT] = pd.to_numeric(results[GLOBAL_STORE_THROUGHPUT], errors = "coerce")

    results[GLOBAL_LOAD_THROUGHPUT] = results[GLOBAL_LOAD_THROUGHPUT].str.replace(",", "")
    results[GLOBAL_LOAD_THROUGHPUT] = pd.to_numeric(results[GLOBAL_LOAD_THROUGHPUT], errors = "coerce")

printStatistics(size, "WEE", results[WARP_EXECUTION_EFFICIENCY])
printStatistics(size, "AO", results[ACHIEVED_OCCUPANCY])
printStatistics(size, "GST", results[GLOBAL_STORE_THROUGHPUT])
printStatistics(size, "GLT", results[GLOBAL_LOAD_THROUGHPUT])
