from typing import List, Set, Tuple
from util import readPrograms

def getFieldTypeLine(field_type, field_name):
    if field_type == "int":
        return f"int {field_name}"
    elif field_type == "bool":
        return f"bool {field_name}"
    elif field_type == "char":
        return f"char {field_name}"
    elif field_type == "float":
        return f"float {field_name}"

def getPrimVal(field_type):
    if field_type == "int":
        return "0"
    elif field_type == "bool":
        return "true"
    elif field_type == "char":
        return "\'a\'"
    elif field_type == "float":
        return "0.0"

def buildFieldAccessExpr(access_type, cls_name, field_name):
    if access_type == "this":
        return "this->"+field_name
    elif access_type == "class_this":
        return cls_name+"::"+field_name
    elif access_type == "super":
        return field_name
    elif access_type == "instance":
        return cls_name+" test; test."+field_name
    elif access_type == "static":
        return cls_name+"::"+field_name
    elif access_type == "simple":
        return field_name

def buildMtdAccessExpr(access_type, cls_name, mtd_name):
    return buildFieldAccessExpr(access_type, cls_name, mtd_name) + "()"

def generateInfo(field_references: Tuple, lang: str) -> str:
    cls_code = ""
    main_code = ""

    SUB_CLASS_NAME = "B"
    METHOD_NAME = "m"
    SUPER_CLASS_NAME = "C"
    QUALIFIED_SUPER_CLASS_NAME = "A.C"
    SUPER_SUPER_CLASS_NAME = "A"
    FIELD_NAME = "field"

    F_MODIFIER_ARR = ["protected", "public"]
    M_MODIFIER_ARR = ["private", "public"]
    ACCESS_TYPE_ARR = ["this", "class_this", "super","instance", "static", "simple"]
    MTD_REL_ARR = ["overload", "unrelated"]
    CLS_REL_ARR = ["inner", "outer", "method"]
    SUPER_CLS_REL_ARR = ["inner", "outer"]

    MTD_ACCESS_TYPE_ARR = ["class_this", "instance", "simple", "this"]

    field_type = "float"
    field_modifier = F_MODIFIER_ARR[field_references[0]]
    method_modifier = M_MODIFIER_ARR[field_references[1]]
    field_access = ACCESS_TYPE_ARR[field_references[2]]
    mtd_rel = MTD_REL_ARR[field_references[3]]
    mtd_access = MTD_ACCESS_TYPE_ARR[field_references[4]]
    subclass_rel = CLS_REL_ARR[field_references[5]]
    superclass_rel = SUPER_CLS_REL_ARR[field_references[6]]

    super_super_cls_decl = "class "+ SUPER_SUPER_CLASS_NAME + "{\n"
    super_super_field_decl = field_type + " " + FIELD_NAME + " = " + getPrimVal(field_type)  

    sub_mtd_decl = "__device__ " if lang == "cuda" else ""
    sub_mtd_decl += "void "+METHOD_NAME+"() {\n"
    sub_mtd_decl += buildFieldAccessExpr(field_access, SUPER_SUPER_CLASS_NAME, FIELD_NAME) + " = " + getPrimVal(field_type) + ";\n"
    sub_mtd_decl += "}\n"

    sub_rel_mtd_decl = "__device__ " if lang == "cuda" else ""
    mtd_name = METHOD_NAME if mtd_rel == "overload" else METHOD_NAME + "2"
    rel_header_overload = "void "+mtd_name+"(int a) {\n"
    rel_header_unrelated = "void "+mtd_name+"() {\n"
    sub_rel_mtd_decl += rel_header_overload if mtd_rel == "overload" else rel_header_unrelated
    sub_rel_mtd_decl += buildMtdAccessExpr(mtd_access, SUB_CLASS_NAME, METHOD_NAME)+";\n"
    sub_rel_mtd_decl += "}\n"

    subclass_prefix = SUPER_SUPER_CLASS_NAME + "::" + SUPER_CLASS_NAME + "::" if subclass_rel == "inner" else ""
    subcls_decl = "class " + subclass_prefix + SUB_CLASS_NAME
    subcls_decl += ": public " + SUPER_CLASS_NAME + " {\n"

    superclass_prefix = SUPER_SUPER_CLASS_NAME + "::" if subclass_rel == "inner" else ""

    super_cls_decl = "class " + superclass_prefix + SUPER_CLASS_NAME
    super_cls_decl += ": public " + SUPER_SUPER_CLASS_NAME + " {\n"
    super_field_decl = field_type + " " + FIELD_NAME + " = " + getPrimVal(field_type)    

    sub_cls_code = subcls_decl
    sub_cls_code += method_modifier+":\n"
    sub_cls_code += sub_mtd_decl
    sub_cls_code += sub_rel_mtd_decl
    sub_cls_code += "};\n"

    super_cls_code = super_cls_decl
    super_cls_code += field_modifier+":\n"
    super_cls_code += super_field_decl+ ";\n" if super_field_decl != "" else ""

    if subclass_rel == "inner":
            super_cls_code += "class " + SUB_CLASS_NAME + ";\n"
    elif subclass_rel == "method":
        super_cls_code += "__device__ " if lang == "cuda" else ""
        super_cls_code += "void classholder() {\n"
        super_cls_code += sub_cls_code
        super_cls_code += "}\n"
    
    super_cls_code += "};\n"

    super_super_cls_code = super_super_cls_decl
    super_super_cls_code += field_modifier+":\n"
    super_super_cls_code += super_super_field_decl+";\n" if super_super_field_decl != "" else ""
    if superclass_rel == "inner":
        super_super_cls_code += "class " + SUPER_CLASS_NAME
        if subclass_rel == "inner":
            super_super_cls_code += "{ class " + SUB_CLASS_NAME+"; }"
        super_super_cls_code += ";\n"
        
    super_super_cls_code += "};\n"

    class_code = super_super_cls_code
    class_code += super_cls_code
    if (subclass_rel == "outer" or subclass_rel == "inner"):
        class_code += sub_cls_code
    
    return class_code


def generateMain(field_references: Tuple, template_file: str, output_file: str, single: bool, lang: str):
    with open(template_file) as template:
        program: List[str] = template.readlines()

    cls_index: int = program.index("CLS HERE\n")

    cls_code = generateInfo(field_references, lang)

    program[cls_index] = f"{cls_code}\n"

    if single:
        return program

    else:
        with open(output_file, 'w') as output:
            for line in program:
                output.write(line)

def threeClsMtdCldGenerate(input_file: str, template_file: str, input_size: int, language: str, output_dir: str, single: bool):
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