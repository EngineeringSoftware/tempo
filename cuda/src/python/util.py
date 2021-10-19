from typing import List, Set, Tuple

# Returns a set of tuples representing the programs.
def readPrograms(file_name: str) -> Set[Tuple]:
    with open(file_name) as results_file:
        results: List[str] = results_file.readlines()
    
    index = results.index("Programs:\n")
    del results[:index + 1]
    
    results = [line.strip() for line in results] # remove newline
    results = [[int(element) for element in graph.split(' ')] for graph in results] # convert elements to integers
    results: Set[Tuple] = set(tuple(graph) for graph in results) # remove duplicates

    return results

def readNestedPrograms(file_name: str) -> List:
    with open(file_name) as results_file:
        results: List[str] = results_file.readlines()
    
    for ind in range(len(results)):
        if "# of tasks created" in results[ind]:
            del results[ind:]
            break
    
    results = [line.strip() for line in results] # remove newline
    results = [line[1:-1] for line in results] # remove first and last []
    results = results[1:]# remove Programs: line 
    final_list = []
    for res in results:
        curr_depths = {0: []}
        depth = 0
        for ch in res:
            if ch == ' ':
                continue
            elif ch == '[':
                depth += 1
                curr_depths[depth] = []
            elif ch == ']':
                depth -= 1
                curr_depths[depth].append(curr_depths[depth+1])
            else:
                curr_depths[depth].append(int(ch))
        final_list.append(curr_depths[0])

    return final_list