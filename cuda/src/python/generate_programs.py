import sys
from cg_generate import cgGenerate
from ig_generate import igGenerate
from nc_generate import ncGenerate
from fs_generate import fsGenerate
from va_generate import vaGenerate
from fr_generate import frGenerate
from cgptr_generate import cgPtrGenerate
from twoclsfldref_generate import twoClsFldRefGenerate
from twoclsfldcld_generate import twoClsFldCldGenerate
from twoclsmtdcld_generate import twoClsMtdCldGenerate
from twoclsmtdpar_generate import twoClsMtdParGenerate
from threeclsmtdcld_generate import threeClsMtdCldGenerate
from ifconstexpr_generate import ifConstExprGenerate
from igcomplex_generate import igComplexGenerate
from igconstructor_generate import igConstructorGenerate
from igoverload_generate import igOverloadGenerate
from igtemplate_generate import igTemplateGenerate
from ptr_generate import ptrGenerate
from expr_generate import exprGenerate

if len(sys.argv) != 10:
    sys.exit("ERROR: " + sys.argv[0] + " file_name + template_file + struct_template + input_type + input_size + language + output_dir + single + n")

input_file: str = sys.argv[1]
template_file: str = sys.argv[2]
struct_template: str = sys.argv[3]
input_type: str = sys.argv[4]
input_size: int = int(sys.argv[5])
language: str = sys.argv[6]
output_dir: str = sys.argv[7]
single: bool = True if sys.argv[8] == "single" else False
n: int = int(sys.argv[9])
print(f"single is {single}")

if input_type == "cg":
    cgGenerate(input_file, template_file, struct_template, input_size, language, output_dir, single)
elif input_type == "ig":
    igGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "expr":
    exprGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "igcomplex":
    igComplexGenerate(input_file, template_file, input_size, language, output_dir, single, n)
elif input_type == "igconstructor":
    igConstructorGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "igoverload":
    igOverloadGenerate(input_file, template_file, input_size, language, output_dir, single, n)
elif input_type == "igtemplate":
    igTemplateGenerate(input_file, template_file, input_size, language, output_dir, single, n)
elif input_type == "ptr":
    ptrGenerate(input_file, template_file, input_size, language, output_dir, single, n)
elif input_type == "nc":
    ncGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "fs":
    fsGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "va":
    vaGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "cgptr":
    cgPtrGenerate(input_file, template_file, struct_template, input_size, language, output_dir, single)
elif input_type == "fr":
    frGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "twoclsfldref":
    twoClsFldRefGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "twoclsfldcld":
    twoClsFldCldGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "twoclsmtdcld":
    twoClsMtdCldGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "twoclsmtdpar":
    twoClsMtdParGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "threeclsmtdcld":
    threeClsMtdCldGenerate(input_file, template_file, input_size, language, output_dir, single)
elif input_type == "ifconstexpr":
    ifConstExprGenerate(input_file, template_file, input_size, language, output_dir, single)
