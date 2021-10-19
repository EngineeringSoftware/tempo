import subprocess
import pathlib
import math  
from random import randint
from typing import List, Set, Tuple
from util import readPrograms
import sys

class Function:
    modifiers: List[str]
    return_type: str
    name: str
    arguments: List[str]
    definition: List[str]
    is_method: bool

    def __init__(self, modifiers: List[str], return_type: str, name: str, arguments: List[str], is_method: bool):
        self.modifiers = modifiers.copy()
        self.return_type = return_type
        self.name = name
        self.arguments = arguments.copy()
        self.is_method = is_method
        if is_method == True:
            self.modifiers.insert(0, "static")

    def get_classname(self):
        return "c" + self.name 

    def start_definition(self):
        if self.is_method == True:
            self.definition = [" ".join(self.modifiers[1:]) + " " + self.return_type + " " + self.get_classname() + "::" + self.name + "(" + ", ".join(self.arguments) + ") {"]
        else:
            self.definition = [self.get_signature()[:-1] + " {"]

    def get_signature(self) -> str:
        signature: str = " ".join(self.modifiers) + " " + self.return_type + " " + self.name + "(" + ", ".join(self.arguments) + ");"
        if self.is_method == True:
            signature = "class " + self.get_classname() + " {\npublic:\n    " + signature + "\n};"
        return signature
        
    def get_ptr(self) -> str:
        return "&" + self.get_classname() + "::" + self.name

    def get_ptr_arg(self, arg_name: str) -> str:
        return self.return_type + " (*" + arg_name + ")(" + ", ".join(self.arguments) + ")"

    def get_ptr_call(self, address_of: bool, ptr_name: str) -> str:
        call_header: str = self.get_classname() + "::" + self.name
        return self.get_call(address_of).replace(call_header, "(*" + ptr_name + ")")

    def get_call(self, address_of: bool) -> str:
        if self.is_method == True:
            call: str = self.get_classname() + "::"
        call += self.name + "( "
        for argument in self.arguments:
            if address_of == True:
                call += "&"
            call += argument.split("*")[-1] + ","

        call = call[:-1]
        return call + ");"

class Variable:
    modifiers: List[str]
    type_name: str
    name: str
    initial_value: str

    def __init__(self, modifiers: List[str], type_name: str, name: str, initial_value: str):
        self.modifiers = modifiers
        self.type_name = type_name
        self.name = name
        self.initial_value = initial_value

    def get_initialization(self):
        return " ".join(self.modifiers) + " " + self.type_name + " " + self.name + " = " + self.initial_value + ";"

    def get_extern_declaration(self):
        return "extern " + " ".join(self.modifiers) + " " + self.type_name + " " + self.name + ";"

    def get_increment(self):
        return self.name + "++;"

STRUCT_NAME: str = "CG"
STRUCTS_HEADER_FILE: str = "struct.h"
STRUCT_NODE_NAME = "Node"

FUNCTION_TYPE: str = "int"
FUNCTION_RECURSION_LIMIT: str = "10"
FUNCTION_NUMBER_OF_ARGUMENTS: int = 3
FUNCTIONS_HEADER_FILE: str = "functions.h"
FUNCTION_IS_METHOD: bool = True

VARIABLE_TYPE: str = "int"
VARIABLE_INITIAL_VALUE: str = "0"

NUM_BLOCKS: str = "1"
NUM_THREADS: str = "1"

# Returns a list of functions.
def createFunctions(cg_size: int, number_of_arguments: int, argument_type: str, modifiers: List[str], type: str, is_method: bool) -> List[Function]:
    function_signatures: List[Function] = [Function(modifiers, type, "f" + str(i), [argument_type + "*p" + str(j) for j in range(number_of_arguments)], is_method) for i in range(cg_size)]
    return function_signatures

# Returns a list of variables.
def createVariables(cg_size: int, modifiers: str, type: str, initial_value: str) -> List[Variable]:
    variables: List[Variable] = [Variable(modifiers, "int", "a" + str(i), "0") for i in range(cg_size)]
    return variables

# Writes the struct definition to a header file
def generateStructHeaderFile(structs_header: str, output_file: str):
    with open(structs_header) as header:
        program: List[str] = header.readlines()

    with open(output_file, 'w') as output:
        for line in program:
            output.write(line + "\n")


# Writes the functions to a single header file.
def generateFunctionHeaderFile(functions: List[Function], variables: List[Variable], structs_header: str, output_file: str, single: bool):
    if not single:
        header: List[str] = ["#ifndef FUNCTIONS_H", "#define FUNCTIONS_H", "#include \"" + structs_header + "\"", "\n"]
    else:
        header = []

    header += [f.get_signature() for f in functions]
    
    if not single:
        header.append("\n")
        header.append("#endif")

    with open(output_file, 'w') as output:
        for line in header:
            output.write(line + "\n")


# Generates the function definitions.
def generateFunctionDefinitions(
    call_graph: Tuple, cg_size: int, functions: List[Function], variables: List[Variable],
    structs_header: str, functions_header: str, recursion_limit: str) -> List[str]:

    source: List[str] = []
    for i in range(cg_size):
        source.append(variables[i].get_initialization())
    # for every function (row) in the callgraph (matrix)
    for i in range(cg_size):
        return_value = " + ".join([arg.split("*")[-1] + "->root->function_id" for arg in functions[i].arguments])
        return_value += " + " + variables[i].get_increment()[:-1]
        return_value += " + answer"

        functions[i].start_definition()
        functions[i].definition.append("    int answer = 0;") 
        functions[i].definition.append("    " + variables[i].get_increment()) 
        functions[i].definition.append("    if (" + variables[i].name + " >= " + recursion_limit +") {")
        functions[i].definition.append("        return " + return_value + ";")
        functions[i].definition.append("    }")

        offset: int = i * cg_size

        # for every element in each row
        for j in range(cg_size):
            # for every call to function j
            for k in range(call_graph[offset + j]):
                functions[i].definition.append("    answer += " + functions[j].get_call(False))
    
        functions[i].definition.append("    return answer;")
        functions[i].definition.append("}")

        source += functions[i].definition

    return source

def initializeArguments(function: Function, argument_instances: List[Tuple]) -> List[str]:
    initialized_arguments: List[str] = [arg.replace("*", " ") + ";" for arg in function.arguments] # Remove pointer type

    struct_sizes: int = int(math.sqrt(len(argument_instances[0])))
    node_pools: List[str] = ["Node pool" + str(x) + "[" + str(struct_sizes) + "];" for x in range(0, len(function.arguments))]
    
    struct_names: List[str] = [arg.split("*")[-1] for arg in function.arguments]
    node_pool_names: List[str] = [pool.split(" ")[-1].split("[")[0] for pool in node_pools]

    adj_matrices: List[str] = ["".join(map(str, matrix)) for matrix in argument_instances]
    init_calls: List[str] = ["CGInit(&" + struct_names[i] + "," + node_pool_names[i] + ",\"" + adj_matrices[randint(0, len(adj_matrices) - 1)] + "\"," + str(struct_sizes) + ");" for i in range(0, len(function.arguments))]

    return init_calls + initialized_arguments + node_pools

def generateMain(
    functions: List[Function], variables: List[Variable], call_graph: Tuple, input_size: int,
    structs_header: str, struct_name: str, functions_header: str, num_blocks: str,
    num_threads: str, argument_instances: List[Tuple], template_file: str, language: str, output_file: str,
    single: bool):
    with open(template_file) as template:
        program: List[str] = template.readlines()

    functions_include: int = program.index("#include \"" + functions_header + "\"\n")
    # program.insert(functions_include - 1, "#include \"" + structs_header + "\"")
    functions_include = program.index("#include \"" + functions_header + "\"\n")
    program.insert(functions_include + 1, "\n".join(generateFunctionDefinitions(call_graph, input_size, functions, variables, FUNCTIONS_HEADER_FILE, STRUCTS_HEADER_FILE, FUNCTION_RECURSION_LIMIT)))

    new_fn_ind: int = program.index("NEW_FN_HERE\n")
    
    if language == "cpp":
        program[new_fn_ind] = "void getRes(" + ", ".join([fn.get_ptr_arg("f"+str(ind)) for ind, fn in enumerate(functions)]) + ") {\n"
        program.insert(new_fn_ind + 1, "}")

        for ind, function in enumerate(functions):
            program.insert(new_fn_ind + 1, "    printf(\"Function " + function.name + " returned %d\\n\", x);")
            program.insert(new_fn_ind + 1, "    x = " + function.get_ptr_call(True, "f"+str(ind)))
        program.insert(new_fn_ind + 1, "    int x;")

        kernel_launch: int = program.index("    test();\n")

        device_variable_copy = "    int y;\n"
        for variable in variables:
            device_variable_copy += "    y = " + variable.name + ";\n"
            device_variable_copy += "    printf(\"%d\\n\", y);\n"
        program.insert(kernel_launch + 1, device_variable_copy)

        kernel_definition: int = program.index("void test() {\n")
        invocation = "getRes(" + ", ".join([fn.get_ptr() for ind, fn in enumerate(functions)]) + ");"
        program.insert(kernel_definition + 1, invocation)


    elif language == "cuda":
        program[new_fn_ind] = "__device__ void get_res(" + ", ".join([fn.get_ptr_arg("f"+str(ind)) for ind, fn in enumerate(functions)]) + ") {\n"
        program.insert(new_fn_ind + 1, "}")

        for ind, function in enumerate(functions):
            program.insert(new_fn_ind + 1, "    printf(\"Function " + function.name + " returned %d\\n\", x);")
            program.insert(new_fn_ind + 1, "    x = " + function.get_ptr_call(True, "f"+str(ind)))
        program.insert(new_fn_ind + 1, "    int x;")

        kernel_launch: int = program.index("    test_kernel<<<B,T>>>();\n")
        program[kernel_launch] = program[kernel_launch].replace('B', num_blocks)
        program[kernel_launch] = program[kernel_launch].replace('T', num_threads)

        device_variable_copy = "    int y;\n"
        for variable in variables:
            device_variable_copy += "    cudaMemcpyFromSymbol(&y, " + variable.name + ", sizeof(" + variable.type_name + "), 0, cudaMemcpyDeviceToHost);\n"
            device_variable_copy += "    printf(\"%d\\n\", y);\n"
        program.insert(kernel_launch + 2, device_variable_copy) # + 2 because of the call to cudaDeviceSynchronize

        kernel_definition: int = program.index("__global__ void test_kernel() {\n")
        invocation = "get_res(" + ", ".join([fn.get_ptr() for ind, fn in enumerate(functions)]) + ");"
        program.insert(kernel_definition + 1, invocation)

    initialized_arguments: List[str] = initializeArguments(functions[0], argument_instances) # we're assuming they all have the same arguments
    for argument in initialized_arguments:
        program.insert(new_fn_ind + 1, "    " + argument)

    if single:
        return program
    else:
        with open(output_file, "w") as output:
            for line in program:
                output.write(line + "\n")

def cgPtrGenerate(input_file: str, template_file: str, struct_template: str, input_size: int, language: str, output_dir: str, single: bool):
    if language == "cpp":
        extension: str = ".cpp"
        FUNCTION_MODIFIERS: List[str] = []
        VARIABLE_MODIFIERS: List[str] = []
    elif language == "cuda":
        extension: str = ".cu"
        FUNCTION_MODIFIERS: List[str] = ["__device__"]
        VARIABLE_MODIFIERS: List[str] = ["__device__"]

    programs: Set[Tuple] = readPrograms(input_file)
    functions: List[Function] = createFunctions(input_size, FUNCTION_NUMBER_OF_ARGUMENTS, STRUCT_NAME, FUNCTION_MODIFIERS, FUNCTION_TYPE, FUNCTION_IS_METHOD)
    variables: List[Variable] = createVariables(input_size, VARIABLE_MODIFIERS, VARIABLE_TYPE, VARIABLE_INITIAL_VALUE)

    generateStructHeaderFile(struct_template, output_dir + "/" + STRUCTS_HEADER_FILE)
    generateFunctionHeaderFile(functions, variables, STRUCTS_HEADER_FILE, output_dir + "/" + FUNCTIONS_HEADER_FILE, single)

    if single:
        output_file: str = output_dir + "/single" + extension
        program_source: List[str] = []
        for idx, cg in enumerate(programs):
            current_program: List[str] = generateMain(
                functions, variables, cg, input_size, STRUCTS_HEADER_FILE,
                STRUCT_NAME, FUNCTIONS_HEADER_FILE, NUM_BLOCKS, NUM_THREADS,
                list(programs), template_file, language, output_file, single)
            

            print("current program")
            print(current_program)
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
        for idx, cg in enumerate(programs):
            output_file: str = output_dir + "/" + str(idx) + extension
            generateMain(functions, variables, cg, input_size, STRUCTS_HEADER_FILE, STRUCT_NAME, FUNCTIONS_HEADER_FILE, NUM_BLOCKS, NUM_THREADS, list(programs), template_file, language, output_file, single)