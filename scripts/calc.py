
def func(filename):
    num = 0;
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            try:
                hits = int(parts[4])
                if hits > 500:
                    num += 500
                else:
                    num += hits
                #num += hits > 500 ? 500 : hits
            except ValueError:
                print(f"err line {line.strip()}")
    return num
    #print(f"total num: {num}")


filenames = ["SMEM.76bp.20000", "SMEM.100bp.20000", "SMEM.152bp.20000","SMEM.251bp.20000"]   
#for filename in filenames:
    #func(filename)
tot_acc = 1
for length in [76, 100, 152, 251]:
    smems = func("SMEM."+str(length)+"bp.20000")
    intvs = func("INTV."+str(length)+"bp.20000")
    acc = smems / intvs
    tot_acc *= acc
    print(f"{length}bp: accuracy {acc * 100} %")

print(f"geomean: accuracy {tot_acc ** 0.25 * 100} %")
