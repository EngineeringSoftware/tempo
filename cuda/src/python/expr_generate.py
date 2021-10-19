from typing import List, Set, Tuple
from util import readNestedPrograms

class Node:
    def getDecl(self):
        return ""
    
    def toStr(self):
        return ""

class PrimaryExpr(Node):
    is_lit: bool
    is_this: bool
    is_id: bool
    is_lambda: bool
    id: int

    def __init__(self, id: int, is_lit: bool, is_this: bool, is_id: bool, is_lambda: bool):
        super().__init__()
        self.is_lit = is_lit
        self.is_this = is_this
        self.is_id = is_id
        self.is_lambda = is_lambda
        self.id = id
    
    def getDecl(self):
        if self.is_id:
            return f"int v{self.id} = 0"
        return ""

    def toStr(self):
        if self.is_lit:
            return "1.0"
        if self.is_this:
            return "this"
        if self.is_lambda:
            return "[](auto a, auto b) { return a < b; }"
        if self.is_id:
            return f"v{self.id}"
            
class InitializerList(Node):
    length: int
    clauses: List[PrimaryExpr]

    def __init__(self, id: int, length: int, clauses: List[PrimaryExpr]):
        super().__init__()
        self.length = length
        self.clauses = clauses
    
    def getDecl(self):
        return ";\n".join([c.getDecl() for c in self.clauses if c.getDecl() != ""])

    def toStr(self):
        return ", ".join([c.toStr() for c in self.clauses])

class AssignmentExpr(Node):
    ASSIGNMENT_OPS = ["="]

    is_primary: bool
    is_init_list: bool
    prim_expr: PrimaryExpr
    op: str
    init_list: InitializerList
    def __init__(self, id: int, is_primary: bool, is_init_list: bool, prim_expr: PrimaryExpr, op: str, init_list: InitializerList):
        super().__init__()
        self.is_primary = is_primary
        self.is_init_list = is_init_list
        self.prim_expr = prim_expr
        self.op = self.ASSIGNMENT_OPS[op]
        self.init_list = init_list
    
    def getDecl(self):
        if self.is_primary:
            return self.prim_expr.getDecl()
        if self.is_init_list:
            return "\n".join([self.prim_expr.getDecl(), self.init_list.getDecl()])


    def toStr(self):
        if self.is_primary:
            return self.prim_expr.toStr()
        if self.is_init_list:
            return self.prim_expr.toStr() + self.op + self.init_list.toStr()


curr_id = 0

def processNode(node: List):
    global curr_id
    node_type = node[0]
    print(node)
    curr_id += 1
    if node_type == 0:
        return PrimaryExpr(curr_id, node[1] == 1, node[2] == 1, node[3] == 1, node[4] == 1)
    if node_type == 2:
        length = node[1]
        nodes = []
        for i in range(2, 2+length):
            nodes.append(processNode(node[i]))
        return InitializerList(curr_id, node[1], nodes)
    if node_type == 3:
        is_prim_expr = node[1] == 1
        is_init_list = node[2] == 1
        if is_prim_expr:
            return AssignmentExpr(curr_id, is_prim_expr, is_init_list, processNode(node[3]), 0, None)
        if is_init_list:
            return AssignmentExpr(curr_id, is_prim_expr, is_init_list, processNode(node[3]), node[4], processNode(node[5]))

NUM_BLOCKS: str = "1"
NUM_THREADS: str = "1"

def generateMain(
    node_arr: List, num_blocks: str, num_threads: str, template_file: str, language: str, output_file: str,
    single: bool):
    with open(template_file) as template:
        program: List[str] = template.readlines()

    if language == "cpp":
        kernel_definition: int = program.index("void test() {\n")

    elif language == "cuda":
        kernel_launch: int = program.index("    test_kernel<<<B,T>>>();\n")
        program[kernel_launch] = program[kernel_launch].replace('B', num_blocks)
        program[kernel_launch] = program[kernel_launch].replace('T', num_threads)

        kernel_definition: int = program.index("__global__ void test_kernel() {\n")
    
    node = processNode(node_arr)
    str_node = node.toStr() 
    str_decl = node.getDecl()

    str_node = str_node + ";\n" if str_node != "" else ""
    str_decl = str_decl + ";\n" if str_decl != "" else ""

    program.insert(kernel_definition + 1, str_node)
    program.insert(kernel_definition + 1, str_decl)

    if single:
        return program
    else:
        with open(output_file, "w") as output:
            for line in program:
                output.write(line + "\n")

def exprGenerate(input_file: str, template_file: str, input_size: int, language: str, output_dir: str, single: bool):
    if language == "cpp":
        extension: str = ".cpp"
    elif language == "cuda":
        extension: str = ".cu"

    programs: List = readNestedPrograms(input_file)

    if single:
        output_file: str = output_dir + "/single" + extension
        program_source: List[str] = []
        for idx, pg in enumerate(programs):
            current_program: List[str] = generateMain(
                pg, NUM_BLOCKS,
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
        for idx, pg in enumerate(programs):
            output_file: str = output_dir + "/" + str(idx) + extension
            generateMain(pg, NUM_BLOCKS, NUM_THREADS, template_file, language, output_file, single)