from typing import List, Set, Tuple
from util import readPrograms

class Function:
    modifiers: List[str]
    is_noexcept: bool
    contains_ifdef: bool

    def __init__(self, modifiers: List[str], is_noexcept: bool, contains_ifdef: bool):
        self.modifiers = modifiers
        self.is_noexcept = is_noexcept
        self.contains_ifdef = contains_ifdef

    def getDeclaration(self) -> str:
        function = " ".join(list(reversed(self.modifiers))) + " void f1() "
        if self.is_noexcept == True:
            function += "noexcept "
        
        function += "{\n"
        if self.contains_ifdef == True:
            function += "#ifdef __CUDA_ARCH__\n"
            function += "f0();\n"
            function += "#endif\n"
        else:
            function += "f0();\n"
        
        function += "}\n"
        return function

def generateFunction(function_specifiers: Tuple, language: str) -> Function:
    modifiers: List[str] = []
    is_noexcept: bool = False
    contains_ifdef: bool = False

    for s in function_specifiers:
        if s == 0:
            modifiers.append("static")
        if s == 1:
            modifiers.append("extern")
        if s == 2:
            modifiers.append("constexpr")
        if s == 3 and language == "cuda":
            modifiers.append(" ")
        if s == 4 and language == "cuda":
            modifiers.append("__device__")
        if s == 5 and language == "cuda":
            modifiers.append("__forceinline__")
        if s == 6 and language == "cuda":
            modifiers.append("__noinline__")
        if s == 7:
            modifiers.append("inline")
        if s == 8:
            modifiers.append("volatile")
        if s == 9:
            is_noexcept = True
        if s == 10:
            contains_ifdef = True

    function: Function = Function(modifiers, is_noexcept, contains_ifdef)
    
    return function


def generateMain(function_specifiers: Tuple, language: str, template_file: str, output_file: str, single: bool):
    with open(template_file) as template:
        program: List[str] = template.readlines()

    function: Function = generateFunction(function_specifiers, language)
    function_index: int = program.index("FUNCTION_HERE\n")
    program[function_index] = function.getDeclaration()

    if single:
        return program
    else:
        with open(output_file, 'w') as output:
            for line in program:
                output.write(line + "\n")

def fsGenerate(input_file: str, template_file: str, input_size: int, language: str, output_dir: str, single: bool):
    if language == "cpp":
        extension: str = ".cpp"
    elif language == "cuda":
        extension: str = ".cu"

    programs: Set[Tuple] = readPrograms(input_file)
    if single:
        output_file: str = output_dir + "/single" + extension
        program_source: List[str] = []

        for idx, fs in enumerate(programs):
            current_program: List[str] = generateMain(fs, language, template_file, output_file, single)

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
        for idx, fs in enumerate(programs):
            output_file: str = output_dir + "/" + str(idx) + extension
            generateMain(fs, language, template_file, output_file)