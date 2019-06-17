import os
import subprocess
import regex as re

cache = [0, 1] #'Data', 'Instruction'
sets = [16, 32, 64, 128, 256, 512]
No_of_sets=[1, 2, 4, 8]
block_size=  16

data_same_data = """
-cache:il1 none
-cache:il2 none
-cache:dl2 none
-tlb:itlb  none
-tlb:dtlb  none
"""

instruction_same_data="""
-cache:il2 none
-cache:dl1 none
-cache:dl2 none
-tlb:itlb none
-tlb:dtlb none
"""

data_miss_ratio_dir = "data_miss_ratio.txt"
ins_miss_ratio_dir = "ins_miss_ratio.txt"
f1 = open("./results/" + data_miss_ratio_dir, "w+")
f2 = open("./results/" + ins_miss_ratio_dir, "w+")

data_miss_dir = "data_miss.txt"
ins_miss_dir = "ins_miss.txt"
f3 = open("./results/" + data_miss_dir, "w+")
f4 = open("./results/" + ins_miss_dir, "w+")

data_acess_dir = "data_acess.txt"
ins_acess_dir = "ins_acess.txt"
f5 = open("./results/" + data_acess_dir, "w+")
f6 = open("./results/" + ins_acess_dir, "w+")



for s in sets:
    data_list = []
    ins_list = []
    for no in No_of_sets:
        # 1. generate config file
        temp_data = str(s)+":16:"+str(no)+":l"
        instruction_data = "-cache:il1 i11:"
        data_data = "-cache:dl1 d11:"

        instruction = instruction_data + temp_data + instruction_same_data
        data = data_data + temp_data + data_same_data
        # print(data)
        data_config_name = "cache_data_sets_" + str(s) + "_Noofsets_" + str(no) + "_block_size_16"
        instruction_config_name = "cache_instruction_sets_" + str(s) + "_Noofsets_" + str(no) + "_block_size_16"
        # print(data_config_name)

        # 2.save config file
        if not os.path.exists("./cache_config"):
            os.mkdir("./cache_config")
        with open("./cache_config/" + data_config_name, "w+") as f:
            f.write(data) 
        with open("./cache_config/" + instruction_config_name, "w+") as f:
            f.write(instruction)            

        # 3.save results
        if not os.path.exists("./results"):
            os.mkdir("./results")
        data_result_dir = data_config_name+".txt"
        instruction_result_dir = instruction_config_name + ".txt"
        data_command = ["./sim-cache", 
                        "-config", 
                        "./cache_config/"+str(data_config_name),
                        "./tests-pisa/bin.little/test-math"]
        # os.system(data_command)
        data_results = subprocess.Popen(data_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        data_stdout, data_stderr = data_results.communicate()
        data_result_data = data_stdout.decode('utf-8')

        ins_command = ["./sim-cache",
                        "-config",
                        "./cache_config/" + str(instruction_config_name),
                        "./tests-pisa/bin.little/test-math"]
        # os.system(ins_command)
        ins_results = subprocess.Popen(ins_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ins_stdout, int_stderr = ins_results.communicate()
        ins_result_data = ins_stdout.decode('utf-8')
        # print(ins_result_data)
        with open("./results/tmp/" + data_result_dir, "w+") as f:
            f.write(data_result_data) 

        with open("./results/tmp/" + instruction_result_dir, "w+") as f:
            f.write(ins_result_data)            

        # miss rate
        pattern_data = "d11.miss_rate.*#"
        pattern_ins = "i11.miss_rate.*#"
        data = re.search(pattern_data, data_result_data)
        ins = re.search(pattern_ins, ins_result_data)
        # miss
        pattern_data_miss = "d11.misses.*#"
        pattern_ins_miss = "i11.misses.*#"
        data_miss = re.search(pattern_data_miss, data_result_data)
        ins_miss = re.search(pattern_ins_miss, ins_result_data)
        # acess
        pattern_data_acess = "d11.accesses.*#"
        pattern_ins_acess = "i11.accesses.*#"
        data_acess = re.search(pattern_data_acess, data_result_data)
        ins_acess = re.search(pattern_ins_acess, ins_result_data)

        # further pattern
        pattern_miss_rate = "[0-9]\.[0-9]{4}"
        pattern_miss = "([0-9]{3,6})"
        pattern_acess = "([0-9]{3,6})"
        # miss rate
        data = re.search(pattern_miss_rate, data[0].replace(' ', ''))
        ins = re.search(pattern_miss_rate, ins[0].replace(' ', ''))
        # miss
        data_miss = re.search(pattern_miss, data_miss[0].replace(' ', ''))
        ins_miss = re.search(pattern_miss, ins_miss[0].replace(' ', ''))
        # acess
        data_acess = re.search(pattern_acess, data_acess[0].replace(' ', ''))
        ins_acess = re.search(pattern_acess, ins_acess[0].replace(' ', ''))

        temp = str(s) + " " + str(no)+" " + str(block_size) + " "
        #f1.write(temp + data[0]+"\n")
        #f2.write(temp + ins[0]+"\n")
        f1.write(data[0]+" ")
        f2.write(ins[0]+" ")
        f3.write(data_miss[0]+" ")
        f4.write(ins_miss[0]+" ")
        f5.write(data_acess[0]+" ")
        f6.write(ins_acess[0]+" ")
        
    f1.write("\n")
    f2.write("\n")
    f3.write("\n")
    f4.write("\n")
    f5.write("\n")
    f6.write("\n")

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()

