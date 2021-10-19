#include <stdio.h>

#include "ptr.h"
#include "../explore.h"

ImpObjPool(Node);

#define OUTPUT_BUFFER 1000000000

__device__ int8_t adjacency_matrices[OUTPUT_BUFFER];

// Offset for adjacency_matrices. If update is set to TRUE,
// then offset will be updated at kernel launch.
__device__ int32_t offset = 0;

// The maximum thread id that successfully generated an inheritance
// graph during the last kernel invocation.
__device__ int32_t max_id = 0;

// Is set to true if a single thread was able to generate
// an inheritance graph.
__device__ int8_t update = FALSE;

void printPgs(int8_t *callgraphs, int valid_cgs, int adj_matrix_size) {
    printf("Programs:\n");
    for (int i = 0; i < valid_cgs; i++) {
        for (int j = 0; j < adj_matrix_size; j++) {
            printf("%d ", callgraphs[(i * adj_matrix_size) + j]);
        }
        printf("\n");
    }
}

void extractPgs(int8_t *source, int8_t *destination, int num_of_cgs, int matrix_size) {
    int destination_index = 0;

    // iterate over each adjacency matrix
    for (int i = 0; i < num_of_cgs; i++) {
        int source_index = i * (matrix_size + 1);
        if (source[source_index] == FALSE) {
            continue;
        }

        memcpy(&destination[destination_index], &source[source_index + 1], matrix_size * sizeof(int8_t));
        destination_index += matrix_size;
    }
}

__device__ int8_t nodeProperty(Node *n, LinkedList *path, Set *visited) {
    assert(path->size == 0);

    LinkedList work;
    llInit(&work);
    llAdd(&work, n);

    while (work.size != 0) {
        Node *current = llRemoveLast(&work);
        if (current == NULL) {
            llRemoveLast(path);
            continue;
        }

        // if not acyclic
        if (llContains(path, current)) {
            return FALSE;
        }

        llAdd(path, current);
        llAdd(&work, NULL);
        setAdd(visited, current);

        // check for diamond
        for (int32_t i = 0; i < current->num_of_children; i++) {
            Node *child = current->children[i];
            for (int32_t j = 0; j < i; j++) {
                if (child == current->children[j]) {
                    return FALSE;
                }
            }
        }

        // add all children
        for (int32_t i = 0; i < current->num_of_children; i++) {
            if (current->children[i] != NULL) {
                llAdd(&work, current->children[i]);
            }
        }
    }

    return TRUE;
}

__device__ int8_t nodePropertyBuggy(Node *n, Set *path, Set *visited) {
    if (setContains(path, n)) {
        return FALSE;
    }

    setAdd(path, n);
    setAdd(visited, n);
    for (int32_t i = 0; i < n->num_of_children; i++) {
        Node *child = n->children[i];
        // two children of a DAG cannot be the same object
        for (int32_t j = 0; j < i; j++) {
            if (child == n->children[j]) {
                return FALSE;
            }
        }
        // check property for every child of this node
        if (child != NULL && !(nodePropertyBuggy(child, path, visited))) {
            return FALSE;
        }
    }

    setRemove(path, n);
    return TRUE;
}

__device__ int8_t noDupChildren(Env *env) {
    NodePool *op = env->op;

    for (int32_t i = 0; i < op->size; i++) {
        Node *n = op->getObject(op, i);
        if (n != NULL) {
            for (int32_t i = 0; i < n->num_of_children; i++) {
                Node *child = n->children[i];
                for (int32_t j = 0; j < i; j++) {
                    if (child == n->children[j]) {
                        return FALSE;
                    }
                }
            }
        }
    }

    return TRUE;
}

__device__ void pgPrint(Node *root, char graph[]) {
    LinkedList work;
    llInit(&work);
    llAdd(&work, root);
    int index = 0;

    while (work.size != 0) {
        Node *current = llRemoveLast(&work);
        if (current == NULL) {
            graph[index++] = 'N';
            graph[index++] = '-';
            graph[index++] = '[';
            graph[index++] = ']';
            graph[index++] = '-';
            continue;
        } else {
            graph[index++] = current->id + '0';
            graph[index++] = '-';
            if (current->num_of_children == 0) {
                graph[index++] = '[';
                graph[index++] = ']';
                graph[index++] = '-';
            }
        }

        // add all children
        for (int32_t i = 0; i < current->num_of_children; i++) {
            llAdd(&work, current->children[i]);
        }
    }
    graph[index++] = 'e';
    graph[index] = '\0';
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    printf("Graph of id %d: %s\n", idx, graph);
}

// Adds generated inheritance graph to adjacency_matrices
__device__ void pgAdjMatrix(Env *env) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    NodePool *op = env->op;
    int adj_matrix_size = (op->size * op->size) + 1;
    int starting_index = (offset + idx) * adj_matrix_size;
    adjacency_matrices[starting_index] = TRUE;

    for (int32_t i = 0; i < op->size; i++) {
        Node *n = op->getObject(op, i);
        if (n != NULL) {
            for (int32_t j = 0; j < n->num_of_children; j++) {
                int8_t child_id = 0;
                if (n->children[j] != NULL) {
                    child_id = n->children[j]->id;
                }

                // + 1 because the first element indicates whether a valid structure exists here
                int index = starting_index + (i * op->size) + child_id + 1;
                if (index >= OUTPUT_BUFFER) {
                    printf("ERROR: index %d is out of bounds\n", index);
                    asm("trap;");
                }

                // adjacency_matrices[index] = n->edge_type[j];
                adjacency_matrices[index]++;
            }
        }
    }
}

__device__ void pgGenerate(Env *env) {
    NodePool *op = env->op;

    for (int32_t i = 0; i < op->size; i++) {
        Node *n = op->getObject(op, i);
        if (n != NULL) {
            n->id = i;
            int32_t num_of_children = _choice(0, env->num_of_nodes - 1);
            n->num_of_children = num_of_children;
            for (int32_t j = 0; j < num_of_children; j++) {
                n->children[j] = op->getAny(op);
                // n->edge_type[j] = _choice(1, 3);
            }
        }
    }
}

__global__ void pgUdita(int32_t bck_active, int32_t size) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    if (idx == 0 && update == TRUE) {
        offset = offset + max_id + 1;
        max_id = 0;
        update = FALSE;
    }

    NodePool op;
    initNodePool(&op, size, EXCLUDE_NULL);

    Env env = {
        .num_of_nodes = size,
        .op = &op,
    };

    pgGenerate(&env);
    int8_t no_dups = noDupChildren(&env);
    _countIf(no_dups);
    if (no_dups == TRUE) {
        pgAdjMatrix(&env);
        atomicMax(&max_id, idx);
        update = TRUE;
    }
    // if (is_dag) {
    //     char graph[50];
    //     dagPrint(dag.root, graph);
    // }
}

__global__ void testNodeProperty(int32_t bck_active) {
    LinkedList ll_path;
    Set path;
    setInit(&path, &ll_path);

    LinkedList ll_visited;
    Set visited;
    setInit(&visited, &ll_visited);

    Node node;
    nodePropertyBuggy(&node, &path, &visited);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Incorrect arguments: size\n");
        exit(1);
    }
    int32_t size = atoi(argv[1]);
    int32_t adj_matrix_size = size * size;

    // establish context
    cudaFree(0);

    float time;
    cudaEvent_t start, stop;
    int8_t *result = (int8_t*) calloc(OUTPUT_BUFFER, sizeof(int8_t));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(adjacency_matrices, result, sizeof(int8_t) * OUTPUT_BUFFER, 0, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));

    int* args = (int*) malloc(sizeof(int));
    args[0] = size;

    CUDA_CHECK_RETURN(cudaEventRecord(start));
    explore((void (*)(...)) pgUdita, args, 1);
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("Driver time: %.2lf\n", time);

    int last_structure;
    int last_id;
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&last_structure, offset, sizeof(int32_t), 0, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&last_id, max_id, sizeof(int32_t), 0, cudaMemcpyDeviceToHost));
    
    int result_size = last_structure + last_id + 1;
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(result, adjacency_matrices, sizeof(int8_t) * (result_size) * (adj_matrix_size + 1), 0, cudaMemcpyDeviceToHost));
    
    int valid_pgs = *(_bck_stats->if_counter);
    int8_t *pointer_graphs = (int8_t*) calloc(valid_pgs * adj_matrix_size, sizeof(int8_t));
    extractPgs(result, pointer_graphs, result_size, adj_matrix_size);

    printPgs(pointer_graphs, valid_pgs, adj_matrix_size);

    free(pointer_graphs);
    free(result);

    // the following line if we use cuda-memcheck --leak-check
    cudaDeviceReset();
    free(args);

    return 0;
}
