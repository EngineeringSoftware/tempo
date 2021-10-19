from typing import List, Set, Tuple
from util import readPrograms

class Variable:
    name: str
    references: List
    var_id: int

    def __init__(self, name: str, var_id: int):
        self.name = name
        self.references = []
        self.var_id = var_id
    
    def add_reference(self, ref):
        self.references.append(ref)
    
    def get_type(self):
        if len(self.references) == 0:
            return "int" 
        elif len(self.references) == 1:
            return "intptr"
        else:
            return "intarr"

    def get_definition(self):
        if len(self.references) == 0:
            return "int "+self.name+";\n"
        elif len(self.references) == 1 and self.references[0].name == self.name:
            return "int * "+self.name+" = " + self.references[0].name + ";\n"
        elif len(self.references) == 1:
            return "int * "+self.name+" = &" + self.references[0].name + ";\n"
        else:
            names = []
            
            for ref in self.references:
                if ref.get_type() == "int":
                    names.append("&" + ref.name)
                elif ref.get_type() == "intptr":
                    names.append(ref.name)
                else:
                    names.append(ref.name+"[0]")

            return "int * "+self.name+"["+str(len(self.references))+"] = {" + ", ".join(names) + "};\n"
    

NUM_BLOCKS: str = "1"
NUM_THREADS: str = "1"

def topologicalSortUtil(index: int, visited: List[bool], stack: List[Variable], variables: List[Variable]) -> List[Variable]:
    visited[index] = True

    for v in variables[index].references:
        if visited[v.var_id] == False:
            stack = topologicalSortUtil(v.var_id, visited, stack, variables)
    
    stack.insert(0, variables[index])
    return stack

def topologicalSort(variables: List[Variable]) -> List[Variable]:
    visited: List[bool] = [False] * len(variables)
    stack: List[Variable] = []

    for v in variables:
        if visited[v.var_id] == False:
            stack = topologicalSortUtil(v.var_id, visited, stack, variables)

    return list(reversed(stack))

def generateVariables(ptr_graph: Tuple, ptr_size: int, variables: List[Variable]) -> List[Variable]:
    # for every variable (row) in the pointer graph (matrix)
    for i in range(ptr_size):
        offset: int = i * ptr_size

        # for every element in each row
        for j in range(ptr_size):
            if ptr_graph[offset + j] == 1:
                variables[j].references.append(variables[i])

    return variables

def createVariables(ptr_size: int) -> List[Variable]:
    variables: List[Variable] = [Variable("v"+str(i), i) for i in range(ptr_size)]
    return variables

def generateMain(
    variables: List[Variable], ptr_graph: Tuple, ptr_size: int,
    num_blocks: str, num_threads: str, template_file: str, language: str, output_file: str,
    single: bool):
    with open(template_file) as template:
        program: List[str] = template.readlines()

    variables: List[Variable] = generateVariables(ptr_graph, ptr_size, variables)
    variables = topologicalSort(variables)

    if language == "cpp":
        kernel_definition: int = program.index("void test() {\n")

    elif language == "cuda":
        kernel_launch: int = program.index("    test_kernel<<<B,T>>>();\n")
        program[kernel_launch] = program[kernel_launch].replace('B', num_blocks)
        program[kernel_launch] = program[kernel_launch].replace('T', num_threads)

        kernel_definition: int = program.index("__global__ void test_kernel() {\n")

    for v in reversed(variables):
        program.insert(kernel_definition + 1, v.get_definition())

    if single:
        return program
    else:
        with open(output_file, "w") as output:
            for line in program:
                output.write(line + "\n")

def ptrGenerate(input_file: str, template_file: str, input_size: int, language: str, output_dir: str, single: bool, n: int):

    programs: Set[Tuple] = readPrograms(input_file)
    if language == "cpp":
        extension: str = ".cpp"
    elif language == "cuda":
        extension: str = ".cu"

    if single:
        output_file: str = output_dir + "/single" + extension
        program_source: List[str] = []
        for idx, pg in enumerate(programs):
            variables: List[Variable] = createVariables(input_size)
            current_program: List[str] = generateMain(
                variables, pg, input_size, NUM_BLOCKS,
                NUM_THREADS, template_file, language, output_file, single)

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
        program_source: List[str] = []
        
        pid = 0
        progs_per_file = round(len(programs) / n) if (len(programs) > n and n != 1) else 1

        for idx, pg in enumerate(programs):
            variables: List[Variable] = createVariables(input_size)
            current_program: List[str] = generateMain(
                variables, pg, input_size, NUM_BLOCKS,
                NUM_THREADS, template_file, language, "", True)

            current_program.insert(0, f"namespace n{idx}{{")
            current_program.append("}")
            program_source.extend(current_program)
            
            if (idx + 1) % progs_per_file == 0:
                output_file: str = output_dir + "/" + str(pid) + extension

                program_source.insert(0, "#include <stdio.h>")
                program_source.append("int main(){")
                for ind in range((idx + 1) - progs_per_file, idx + 1):
                    program_source.append(f"n{ind}::main();")
                program_source.append("}")
                with open(output_file, "w") as output:
                    for line in program_source:
                        output.write(line + "\n")

                pid += 1
                program_source: List[str] = []