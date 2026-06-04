import os, csv
base = "evaluation_results/2nd_full_experiment"
results = []
for exp_dir in sorted(os.listdir(base)):
    if not exp_dir.endswith(("_run1","_run2","_run3")):
        continue
    csv_path = os.path.join(base, exp_dir, "k8s_node_resources.csv")
    if not os.path.exists(csv_path):
        results.append((exp_dir, "NO CSV"))
        continue
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        nodes = {}
        ts = None
        for row in reader:
            if ts is None: ts = row["timestamp"]
            if row["timestamp"] == ts:
                cpu = row["cpu_cores"]
                try:
                    val = int(cpu.replace("m","")) if cpu.endswith("m") else int(float(cpu)*1000)
                except:
                    val = 0
                nodes[row["node"]] = val
            else:
                break
    total = sum(nodes.values())
    flag = " *** HOT START ***" if total > 500 else ""
    results.append((exp_dir, f"{total}m{flag}"))
for r in results:
    print(r[0], ":", r[1])
