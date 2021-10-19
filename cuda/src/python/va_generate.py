from typing import List, Set, Tuple
from util import readPrograms

def generateMain(val: int, template_file: str, output_file: str, single: bool):
    with open(template_file) as template:
        program: List[str] = template.readlines()
    print(program)
    va_index: int = program.index("ASSIGNMENT HERE\n")
    program[va_index] = f"x = {val};"

    if single:
        return program
    else:
        with open(output_file, 'w') as output:
            for line in program:
                output.write(line + "\n")

def vaGenerate(input_file: str, template_file: str, input_size: int, language: str, output_dir: str, single: bool):
    if language == "cpp":
        extension: str = ".cpp"
    elif language == "cuda":
        extension: str = ".cu"

    programs: Set[Tuple] = readPrograms(input_file)
    if single:
        output_file: str = output_dir + "/single" + extension
        program_source: List[str] = []

        for idx, vals in enumerate(programs):
            current_program: List[str] = generateMain(vals[0], template_file, output_file, single)

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
        for idx, vals in enumerate(programs):
            output_file: str = output_dir + "/" + str(idx) + extension
            generateMain(vals[0], template_file, output_file, single)