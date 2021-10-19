from typing import List, Set, Tuple
from util import readPrograms

class Class:
    name: str
    methods: List[str]
    field: str

    def __init__(self, name: str, field: str, field_is_static: bool):
        self.name = name
        self.inner_class = None
        self.methods = []
        self.field = "int " + field + ";"
        if field_is_static == True:
            self.field = "static " + self.field

    def get_field_name(self) -> str:
        return self.field.split(" ")[-1][:-1]
    
    def getDeclaration(self, spaces: str = "") -> str:
        declaration: str = spaces + "class " + self.name + " {\n"
        declaration += spaces + "public:\n"
        declaration += spaces + "    " + self.field + "\n"

        if self.inner_class != None:
            declaration += self.inner_class.getDeclaration(spaces + "    ") 
        
        self.methods = [spaces + "    " + m for m in self.methods]
        declaration += "\n".join(self.methods)
        declaration += "\n" + spaces + "};\n"
        return declaration


def generateAccessingMethods(classes: List[Class], access_location: int, method_is_static: bool, access_operator: str, initialization_operator: str, class_accessed: int):
    method: str = "void m() {"
    if method_is_static == True:
        method = "static " + method

    method += initialization_operator.replace("N", classes[class_accessed].name)
    method += access_operator + classes[class_accessed].get_field_name() + " = 8;}" 
    classes[access_location].methods.append(method)

def generateClasses(nested_classes: Tuple) -> List[Class]:
    size: int = nested_classes[0]
    access_location: int = nested_classes[1]

    if nested_classes[2] == 0:
        field_is_static: bool = True
    elif nested_classes[2] == 1:
        field_is_static: bool = False
    
    if nested_classes[3] == 0:
        method_is_static: bool = True
    elif nested_classes[3] == 1:
        method_is_static: bool = False

    if nested_classes[4] == 0:
        access_operator: str = ""
    elif nested_classes[4] == 1:
        access_operator: str = "::"
    elif nested_classes[4] == 2:
        access_operator: str = "."
    elif nested_classes[4] == 3:
        access_operator: str = "->"

    if nested_classes[5] == 0:
        initialization_operator: str = ""
    elif nested_classes[5] == 1:
        initialization_operator: str = "new N()"
    elif nested_classes[5] == 2:
        initialization_operator: str = "N()"
    elif nested_classes[5] == 3:
        initialization_operator: str = "N{}"
    elif nested_classes[5] == 4:
        initialization_operator: str = "N"
    
    class_accessed: int = nested_classes[6]

    classes: List[Class] = [Class("c" + str(i), "f" + str(i), field_is_static) for i in range(size)]
    for i in range(size - 1):
        classes[i].inner_class = classes[i + 1]

    generateAccessingMethods(classes, access_location, method_is_static, access_operator, initialization_operator, class_accessed)
    
    return classes


def generateMain(nested_classes: Tuple, template_file: str, output_file: str, single: bool):
    with open(template_file) as template:
        program: List[str] = template.readlines()

    classes: List[Class] = generateClasses(nested_classes)
    program.insert(0, classes[0].getDeclaration())

    if single:
        return program

    else:
        with open(output_file, 'w') as output:
            for line in program:
                output.write(line + "\n")

def ncGenerate(input_file: str, template_file: str, input_size: int, language: str, output_dir: str, single: bool):
    if language == "cpp":
        extension: str = ".cpp"
    elif language == "cuda":
        extension: str = ".cu"

    programs: Set[Tuple] = readPrograms(input_file)
    if single:
        output_file: str = output_dir + "/single" + extension
        program_source: List[str] = []
        for idx, nc in enumerate(programs):
            current_program: List[str] = generateMain(nc, template_file, output_file, single)

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
            generateMain(nc, template_file, output_file)