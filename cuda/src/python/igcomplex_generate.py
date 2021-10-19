from typing import List, Set, Tuple
from util import readPrograms

class Method:
    modifiers: List[str]
    return_type: str
    name: str
    is_pure_virtual: bool
    arguments: List[str]
    definition: List[str]

    def __init__(self, modifiers: List[str], return_type: str, name: str, arguments: List[str], definition: List[str], is_pure_virtual: bool):
        self.modifiers = modifiers
        self.return_type = return_type
        self.name = name
        self.arguments = arguments
        self.definition = definition
        self.is_pure_virtual = is_pure_virtual

    def get_call(self, arguments: List[str]) -> str:
        call: str = self.name
        call += "("
        
        if len(self.arguments) > 0:
            for arg in arguments:
                call += arg + ","
            call = call[:-1]

        call += ");\n"

        return call

    
    def get_definition(self) -> str:
        definition: str = " ".join(self.modifiers) + " " + self.return_type + " " + self.name + "("

        if len(self.arguments) > 0:
            for arg in self.arguments:
                definition += arg + ","
            definition = definition[:-1]

        if self.is_pure_virtual:
            definition += ") = 0;\n"
        else:
            definition += ") {\n"

            definition += "".join(self.definition) + "\n"
            definition += "}\n"

        return definition


class Field:
    modifiers: List[str]
    type_name: str
    name: str
    initial_value: str

    def __init__(self, modifiers: List[str], type_name: str, name: str, initial_value: str):
        self.modifiers = modifiers
        self.type_name = type_name
        self.name = name
        self.initial_value = initial_value
    
    def get_declaration(self) -> str:
        declaration: str = " ".join(self.modifiers) + " " + self.type_name + " " + self.name
        if self.initial_value != "":
            declaration += " = " + self.initial_value
        
        declaration += ";"

        return declaration
    
class Class:
    name: str
    fields: List[Field]
    methods: List[Method]
    class_id: int

    def __init__(self, method_modifiers: List[str], field_modifiers: List[str], name: str, class_id: int):
        self.name = name
        self.class_id = class_id
        self.fields = [Field(field_modifiers, "int", "x", name[1:])]
        self.methods = [Method(method_modifiers, "void", "print", [], ["    printf(\"x is %d\\n\", x);"], False)]
        self.parents = []
        self.children = []
        self.inheritance_modifiers = ["public"]

    def get_definition(self) -> str:
        definition: str = "struct " + self.name
        if (len(self.parents) > 0):
            definition += " : "
            for parent in self.parents:
                definition += " ".join(self.inheritance_modifiers) + " " + parent.name + ", "
            definition = definition[:-2]

        definition += " {\n"
        definition += "public:\n"

        for f in self.fields:
            definition += "    " + f.get_declaration() + "\n"
        
        for m in self.methods:
            definition += m.get_definition() + "\n"

        definition += "};\n"
        return definition

NUM_BLOCKS: str = "1"
NUM_THREADS: str = "1"

def topologicalSortUtil(index: int, visited: List[bool], stack: List[Class], classes: List[Class]) -> List[Class]:
    visited[index] = True

    for c in classes[index].children:
        if visited[c.class_id] == False:
            stack = topologicalSortUtil(c.class_id, visited, stack, classes)
    
    stack.insert(0, classes[index])
    return stack

def topologicalSort(classes: List[Class]) -> List[Class]:
    visited: List[bool] = [False] * len(classes)
    stack: List[Class] = []

    for c in classes:
        if visited[c.class_id] == False:
            stack = topologicalSortUtil(c.class_id, visited, stack, classes)

    return list(reversed(stack))


def createClasses(method_modifiers: List[str], field_modifiers: List[str], ig_size: int) -> List[Class]:
    class_names: List[Class] = [Class(method_modifiers, field_modifiers, "c" + str(i), i) for i in range(ig_size)]
    return class_names

def generateClasses(inheritance_graph: Tuple, method_modifiers: List[str], ig_size: int, classes: List[Class]) -> List[Class]:
    print(inheritance_graph)
    # for every class (row) in the inheritance graph (matrix)
    for i in range(ig_size):
        offset: int = i * ig_size

        # for every element in each row
        for j in range(ig_size):
            if inheritance_graph[offset + j] == 1:
                classes[j].parents.append(classes[i])
                classes[i].children.append(classes[j])

    INHERITANCE_TYPES = ["public", "protected", "private"]
    curr_ind = ig_size * ig_size
    for i in range(ig_size):
        has_method = inheritance_graph[curr_ind]
        method_type = inheritance_graph[curr_ind + 1]
        has_method_arg = inheritance_graph[curr_ind + 2]
        inheritance_type = INHERITANCE_TYPES[inheritance_graph[curr_ind + 3]]
        is_virtual = inheritance_graph[curr_ind + 4]

        inheritance_modifiers = ["virtual", inheritance_type] if is_virtual == 1 else [inheritance_type]
        classes[i].inheritance_modifiers = inheritance_modifiers

        print("Got has_method: "+str(has_method)+ " method_type: "+str(method_type)+ " has_method_arg: "+str(has_method_arg)+" inheritance_type: "+str(inheritance_type)+" is_virtual: "+str(is_virtual)+"")

        if has_method == 1:
            curr_method_mods = method_modifiers
            curr_args = []
            curr_pure_virtual = method_type == 1
            if method_type == 0 or method_type == 1:
                curr_method_mods = method_modifiers + ["virtual"]
            if has_method_arg == 1:
                curr_args += ["int a"]
            classes[i].methods.append(Method(curr_method_mods, "void", "n", curr_args, ["    printf(\"x = %d\", x);"], curr_pure_virtual))
        curr_ind += 5

    return classes

def generateMain(
    classes: List[Class], inheritance_graph: Tuple, ig_size: int, method_modifiers: List[str],
    num_blocks: str, num_threads: str, template_file: str, language: str, output_file: str,
    single: bool):
    with open(template_file) as template:
        program: List[str] = template.readlines()

    classes: List[Class] = generateClasses(inheritance_graph, method_modifiers, ig_size, classes)
    classes = topologicalSort(classes)

    for c in classes:
        program.insert(1, c.get_definition())

    if language == "cpp":
        kernel_definition: int = program.index("void test() {\n")

    elif language == "cuda":
        kernel_launch: int = program.index("    test_kernel<<<B,T>>>();\n")
        program[kernel_launch] = program[kernel_launch].replace('B', num_blocks)
        program[kernel_launch] = program[kernel_launch].replace('T', num_threads)

        kernel_definition: int = program.index("__global__ void test_kernel() {\n")

    for i in range(len(classes)):
        class_virtual = any(["virtual" in method.modifiers for method in classes[i].methods]) or "virtual" in classes[i].inheritance_modifiers
        has_n = any([method.name == "n" for method in classes[i].methods])
        if not class_virtual:
            for method in classes[i].methods :
                if "virtual" not in method.modifiers:
                    program.insert(kernel_definition + 1, "    o" + str(i) + "." + method.get_call(["1"]))
            
            if not has_n:
                    program.insert(kernel_definition + 1, "    o" + str(i) + ".n(1);")
            program.insert(kernel_definition + 1, "    " + classes[i].name + " o" + str(i) + ";")

    if single:
        return program
    else:
        with open(output_file, "w") as output:
            for line in program:
                output.write(line + "\n")

def igComplexGenerate(input_file: str, template_file: str, input_size: int, language: str, output_dir: str, single: bool, n: int):
    if language == "cpp":
        extension: str = ".cpp"
        METHOD_MODIFIERS: List[str] = []
        FIELD_MODIFIERS: List[str] = []
    elif language == "cuda":
        extension: str = ".cu"
        METHOD_MODIFIERS: List[str] = ["__device__"]
        FIELD_MODIFIERS: List[str] = []

    programs: Set[Tuple] = readPrograms(input_file)

    if single:
        output_file: str = output_dir + "/single" + extension
        program_source: List[str] = []
        for idx, ig in enumerate(programs):
            classes: List[Class] = createClasses(METHOD_MODIFIERS, FIELD_MODIFIERS, input_size)
            current_program: List[str] = generateMain(
                classes, ig, input_size, METHOD_MODIFIERS, NUM_BLOCKS,
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

        for idx, ig in enumerate(programs):
            classes: List[Class] = createClasses(METHOD_MODIFIERS, FIELD_MODIFIERS, input_size)
            current_program: List[str] = generateMain(
                classes, ig, input_size, METHOD_MODIFIERS, NUM_BLOCKS,
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