from typing import List, Set, Tuple
from util import readPrograms
            
STRUCT_SIZE = 1

def generateNesting(cond_arr, instructs_arr):
    nested_lvl = 0
    code = ""

    found_if = False
    for i in range(len(cond_arr)):
        instruct =instructs_arr[i]

        if instruct == "indent" or not found_if:
            code += "if (std::" + cond_arr[i] + "<T>::value) {\n"
            nested_lvl += 1
            found_if = True
        elif instruct == "same":
            code += "return true; } \n else if (std::" + cond_arr[i] + "<T>::value) {\n"
        else:
            code += "return true; } \n else { return false; }\n"
            nested_lvl -= 1
            found_if = False
    
    
    while nested_lvl > 0:
        code += "return true;\n"
        code += "}\n"
        nested_lvl -= 1
    
    return code

def generateMainCode(base_type):
    main_code = ""

    if base_type == "float":
        main_code += "float dummy = 0.9;\n"
    elif base_type == "int":
        main_code += "int dummy = 2;\n"
    elif base_type == "char":
        main_code += "char dummy = 'a';\n"
    
    main_code += "dummy_fn(dummy);"
    return main_code



def generateInfo(branch_instructs: Tuple) -> Tuple[str, str]:
    nested_lvl = 0
    COND_MAPPINGS = ["is_integral", "is_void", "is_floating_point", "is_class", "is_scalar"]
    INSTRUCT_MAPPINGS = ["indent", "same", "dedent"]
    TYPE_MAPPING = ["float", "int", "char"]

    instructs_len = (len(branch_instructs) - STRUCT_SIZE) // 2

    cond_arr = list()
    
    curr_ind = 0
    for i in range(instructs_len):
        cond_arr.append(COND_MAPPINGS[branch_instructs[i]])

    curr_ind += instructs_len
    base_type = TYPE_MAPPING[branch_instructs[curr_ind]]
    curr_ind += 1

    instructs_arr = list()
    for i in range(instructs_len):
        instructs_arr.append(INSTRUCT_MAPPINGS[branch_instructs[curr_ind + i]])

    cls_code = generateNesting(cond_arr, instructs_arr)
    main_code = generateMainCode(base_type)

    return cls_code, main_code


def generateMain(field_reference: Tuple, template_file: str, output_file: str, single: bool, lang: str):
    with open(template_file) as template:
        program: List[str] = template.readlines()

    code_str, main_str = generateInfo(field_reference)
    cls_index: int = program.index("CODE HERE\n")
    body_index: int = program.index("MAIN HERE\n")

    program[cls_index] = f"{code_str}\n"
    program[body_index] = f"{main_str}\n"

    if single:
        return program

    else:
        with open(output_file, 'w') as output:
            for line in program:
                output.write(line)

def ifConstExprGenerate(input_file: str, template_file: str, input_size: int, language: str, output_dir: str, single: bool):
    if language == "cpp":
        extension: str = ".cpp"
    elif language == "cuda":
        extension: str = ".cu"

    programs: Set[Tuple] = readPrograms(input_file)
    if single:
        output_file: str = output_dir + "/single" + extension
        program_source: List[str] = []
        for idx, nc in enumerate(programs):
            current_program: List[str] = generateMain(nc, template_file, output_file, single, language)

            current_program.insert(0, f"namespace n{idx}{{")
            current_program.append("}")
            program_source.extend(current_program)

        program_source.insert(0, "#include <stdio.h>")
        program_source.append("int main(){")
        for idx, cg in enumerate(programs):
            program_source.append(f"n{idx}::main();")
        program_source.append("}")
        with open(output_file, "w") as output:
            for line in program_source:
                output.write(line + "\n")

    else:
        for idx, nc in enumerate(programs):
            output_file: str = output_dir + "/" + str(idx) + extension
            # print(" ".join(str(x) for x in nc) + " is at " + output_file)
            generateMain(nc, template_file, output_file, single, language)