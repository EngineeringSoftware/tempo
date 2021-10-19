from typing import List, Set, Tuple
from util import readPrograms

NUM_VARS = 1
NUM_METHODS = 1
NUM_SUPERCLS = 2

class Variable:
    name: str
    modifier: str

    def __init__(self, name: str, modifier: str):
        self.name = name
        self.modifier = modifier
    
    def var_init(self):
        if self.modifier == "static":
            return f"{self.modifier} int {self.name};"
        return f"{self.modifier} int {self.name} = 0;"
    
    def var_name(self):
        return self.name
    

class MethodDecl:
    name: str
    add_parens: bool
    var_ind: int
    qualifier: str
    same_cls: bool
    # var_cls: ClassDecl

    def __init__(self, name: str, qualifier: str):
        self.name = name
        # self.add_parens = add_parens
        self.var_cls = None
        self.same_cls = False
        self.var_ind = -1
        self.qualifier = qualifier

    def add_var_cls_ind(self, var_cls, var_ind: int, same_cls: bool):
        self.var_cls = var_cls
        self.var_ind = var_ind
        self.same_cls = same_cls

    def mtd_init(self, lang):
        device = ""
        if lang == "cuda":
            device = "__device__"
        return device + " " +str(self.qualifier) + " void " + str(self.name) + "() { "+ str(self.var_cls.get_var_call(self.var_ind, self.same_cls)) +"++; }"
    
    def mtd_call(self, var_name, cls_name):
        if self.qualifier == "static":
            return f"{cls_name}::{self.name}();"
        return f"{var_name}.{self.name}();"


class ClassDecl:
    name: str
    vars: List[Variable]
    mtds: List[MethodDecl]
    # supercls: List[ClassDecl]

    def __init__(self, name, vars: List[Variable], mtds: List[MethodDecl]):
        self.name = name
        self.vars = vars
        self.mtds = mtds
        self.supercls = list()

    def add_supercls(self, cls_name):
        self.supercls.append(cls_name)

    def get_var_call(self, var_ind, same_cls):
        if self.vars[var_ind].modifier == "static":
            name: str = f"{self.name}::" if not same_cls else ""
            return f"{name}{self.vars[var_ind].name}"
        elif same_cls:
            return f"this->{self.vars[var_ind].name}"
        else:
            return f"{self.name} t; t.{self.vars[var_ind].name}"
    
    def cls_init(self, lang):
        res = f"class {self.name} "
        if len(self.supercls) > 0:
            supercls_names = [elem.name for elem in self.supercls]
            res += ": public " + supercls_names[0]

            for i in range(1, len(supercls_names)):
                res += ", public " + supercls_names[i]
        
        res += "{\n"
        res += "public:\n"

        for var in self.vars:
            res += var.var_init() + "\n"

        for mtd in self.mtds:
            res += mtd.mtd_init(lang) + "\n"
        
        res += "};\n\n"
        return res
    
    def get_method_calls(self, name):
        res = f"{self.name} {name};\n"

        for mtd in self.mtds:
            res += mtd.mtd_call(name, self.name) + "\n"
        
        return res
            

def generateInfo(field_reference: Tuple) -> List[ClassDecl]:

    classes = list()
    superclasses = list()
    mtds_info = list()
    MODIFIER_MAP = {0: "", 1: "volatile", 2: "mutable", 3: "static"}
    QUALIFIER_MAP = {0: "", 1: "volatile", 2: "static"}
    print(field_reference)
    NUM_CLASSES = len(field_reference) // (NUM_VARS + NUM_SUPERCLS + 3)
    curr_i = 0

    for cls_ind in range(NUM_CLASSES):
        vars = list()
        for i in range(NUM_VARS):
            vars.append(Variable(f"v{i}", MODIFIER_MAP[field_reference[curr_i + i]]))
        
        curr_i += NUM_VARS

        mtds = list()
        mtds_info_sublist = list()
        for i in range(NUM_METHODS):
            # add_parens = (field_reference[curr_i + i] == 1)
            var_num = field_reference[curr_i + i ]
            print(f"Var num: {var_num}")
            cls_num = field_reference[curr_i + i +1]
            qualifier = QUALIFIER_MAP[field_reference[curr_i + i + 2]]
            mtds.append(MethodDecl(f"m{i}", qualifier))
            mtds_info_sublist.append([var_num, cls_num])
        
        mtds_info.append(mtds_info_sublist)
        
        curr_i += NUM_METHODS * 3
        supercls = list()
        for i in range(NUM_SUPERCLS):
            if field_reference[curr_i + i] not in supercls:
                supercls.append(field_reference[curr_i + i])
        curr_i += NUM_SUPERCLS
        superclasses.append(supercls)
        classes.append(ClassDecl(f"c{cls_ind}", vars, mtds))

    for i in range(len(superclasses)):
        cls_super_list = superclasses[i]
        for cls_super in cls_super_list:
            if cls_super != i:
                classes[i].add_supercls(classes[cls_super])
    
    for i in range(len(mtds_info)):
        mtds_info_list = mtds_info[i]
        curr_ind = 0
        for var_num, cls_num in mtds_info_list:
            classes[i].mtds[curr_ind].add_var_cls_ind(classes[cls_num], var_num, cls_num==i)
            curr_ind += 1

    return classes


def generateMain(field_reference: Tuple, template_file: str, output_file: str, single: bool, lang: str):
    with open(template_file) as template:
        program: List[str] = template.readlines()

    classes: List[ClassDecl] = generateInfo(field_reference)
    cls_index: int = program.index("CLS HERE\n")
    body_index: int = program.index("MAIN BODY HERE\n")

    main_str = ""
    cls_str = ""

    for i in range(len(classes)):
        main_str += classes[i].get_method_calls(f"c{i}_test")
        cls_str += classes[i].cls_init(lang)

    program[cls_index] = f"{cls_str}\n"
    program[body_index] = f"{main_str}\n"

    if single:
        return program

    else:
        with open(output_file, 'w') as output:
            for line in program:
                output.write(line)

def frGenerate(input_file: str, template_file: str, input_size: int, language: str, output_dir: str, single: bool):
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