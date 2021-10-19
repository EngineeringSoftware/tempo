import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

plt.rcParams["axes.labelsize"] = 15
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

if len(sys.argv) != 5:
    sys.exit("ERROR: " + sys.argv[0] + " file_name + output_loc + subject_type + experiment")

csv_file = sys.argv[1]
output_loc = sys.argv[2]
subject_type = sys.argv[3]
experiment = sys.argv[4]
df = pd.read_csv(csv_file)

print(sys.argv, file=sys.stderr)

if experiment != "compiletypes":
    for i, v in enumerate(df["time"]):
        if v == 0:
            df["time"][i] = 1
else:
    grouped_df = df.groupby(['subject','size'])
    ctype_mappings = {"parallel": "Parallel"}
    for group in grouped_df.groups:
        updated_df = grouped_df.get_group(group)
        res = {}
        is_first = False
        for index, row in updated_df.iterrows():
            numgroups  = int(row["numgroups"])
            if row["compiletype"] == "parallel":
                key = ctype_mappings[row["compiletype"]] if numgroups == 1 else ctype_mappings[row["compiletype"]] + " " + str(numgroups)
                res[key] = {}
                res[key]["GCC"] = row["gcc_ratio"]
                res[key]["Clang++"] = row["clang_ratio"]
                res[key]["NVCC"] = row["nvcc_ratio"]
                res[key]["Clang++ (CUDA)"] = row["clang_cuda_ratio"]

                if not is_first:
                    is_first = True
                    res["Generation"] = {}
                    res["Generation"]["GCC"] = row["gcc_gen_imp"]
                    res["Generation"]["Clang++"] = row["clang_gen_imp"]
                    res["Generation"]["NVCC"] = row["nvcc_gen_imp"]
                    res["Generation"]["Clang++ (CUDA)"] = row["clang_cuda_gen_imp"]
        
        print(res)
        pd.DataFrame(res).plot(kind='bar', rot=0)
        # plt.xticks(rotation=90)
        plt.savefig(output_loc+"/compiletimes-"+group[0]+"-"+str(group[1])+".eps", dpi = 300, bbox_inches='tight')
        plt.clf()
    sys.exit(0)

if experiment == "wlsizetest":
    fig, ax = plt.subplots()

    if subject_type == "seq":
        style = "Length"
    elif subject_type == "udita":
        style = "Size"

    df[style] = df["length"]
    df.dropna(inplace = True)
    df[style] = df[style].map(str) + "-" + df["tasks"].map(str)
    lp = sns.lineplot(x="WLsize", y="time", style=style,
                    markers=True, dashes=False, data=df, ax=ax)

    exploretype = df["exploretype"][0]
    minvalue = min(df["WLsize"])
    maxvalue = max(df["WLsize"])
    print(maxvalue)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    if df["exploretype"][0] == "rangethreads":
        xticks = list(df["WLsize"])
        xticks.insert(0, 0)

        xlabels = [0]
        currentvalue = minvalue
        if currentvalue <= 0:
            sys.exit()
        while currentvalue <= maxvalue:
            xlabels.append(currentvalue)
            currentvalue *= 2
        
        xlabels = [int(x / 1000000) for x in xlabels]
        if "ha-udita" in csv_file:
            ylabels = [f"{int(y / 1000)}" for y in lp.get_yticks()]
        else:
            ylabels = ["{:.1f}".format(y / 1000) for y in lp.get_yticks()]
        
        lp.set(xlabel = "Chunk Size (in millions of threads)", ylabel = "Time [s]", xticks = xticks, xticklabels = xlabels, yticklabels = ylabels)

        lp.set_xticklabels(lp.get_xticklabels(), size = 12)

        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        y_offset = (ymax - ymin) / 75
        x_offset = (xmax - xmin) / 75
        
        i = 0
        for x, y, name in zip(list(df["WLsize"]), list(df["time"]), list(df["kernels"])):
            if i == 0:
                ax.text(x - x_offset, y - 5 * y_offset, name, color="g", size = 12)
                i += 1
            else:
                ax.text(x + x_offset, y + y_offset, name, color="g", size = 12)
        
        plt.xticks(rotation=90)

    elif df["exploretype"][0] == "exactthreads":
        lp.set(ylabel = "Time [ms]")
        ax.set(yscale = "log")
        lp.set(xlabel = "Worklist Size")

elif experiment == "timevslength":
    plt.rc('legend', fontsize='large')

    fig, ax = plt.subplots()
    ax.set(yscale = "log")
    df["Algorithm"] = df["exploretype"].map(
        {"allthreads": "ConstStrategy (CUDA)", "exactthreads": "ReexeStrategy (CUDA)", "rangethreads": "ForkStrategy (CUDA)",
        "allthreadsomp": "ForkStrategy (OpenMP)", "exactthreadsomp": "ReexeStrategy (OpenMP)"})

    if subject_type == "seq":
        x = "Length"
    elif subject_type == "udita":
        x = "Size"

    df[x] = df["length"]
    palette = {
        "ForkStrategy (CUDA)" : "#6ACC65", "ReexeStrategy (CUDA)": "#D65F5F", "ConstStrategy (CUDA)": "orange",
        "ForkStrategy (OpenMP)": "purple", "ReexeStrategy (OpenMP)" : "blue"
    }

    legend_order = ["ForkStrategy (CUDA)", "ReexeStrategy (CUDA)", "ForkStrategy (OpenMP)", "ReexeStrategy (OpenMP)"]

    if "ConstStrategy (CUDA)" in df.Algorithm.values:
        legend_order.append("ConstStrategy (CUDA)")

        # Make sure there is no overlap with the plot
        if "dag" in csv_file:
            plt.rc('legend', fontsize='medium')

    lp = sns.lineplot(x=x, y="time", hue="Algorithm", hue_order=legend_order,
                      marker='o', style="Algorithm", data=df, ax=ax, palette=palette)
    
    minvalue = min(df["length"])
    maxvalue = max(df["length"])

    xlabels = []
    for i in range(minvalue, maxvalue + 1):
        xlabels.append(i)

    lp.set(ylabel = "Time [ms]", xticks = xlabels, xticklabels=xlabels)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    lp.set_xticklabels(lp.get_xticklabels(), size = 15)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
else:
    sys.exit("ERROR: INVALID EXPERIMENT")

plot = lp.get_figure()
if experiment == "wlsizetest":
    if df["exploretype"][0] == "rangethreads":
        plot.savefig(output_loc, dpi = 300, bbox_inches='tight')
    else:
        plot.subplots_adjust(bottom=0.15)
        plot.savefig(output_loc, dpi = 300, bbox_inches='tight')
else:
    plot.savefig(output_loc, dpi = 300, bbox_inches='tight')
