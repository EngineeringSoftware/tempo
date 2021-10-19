
#include "dag.h"
#include "../explore.h"
#include "../mains.h"

ImpObjPool(Node);

// Needed for running a test.

// Level is how many nodes to traverse before inserting
// the node. Traversal stops if leaf node is reached.
__device__ void dagAddNode(Env *env, DAG* dag, unsigned int level, Node* n) {
    Node *current = dag->root;
    for (int i = 0; i < level; i++) {
        if (current->children[0] == NULL) {
            break;
        }

        current = current->children[0];
    }

    int num_of_children = current->num_of_children;
    current->children[num_of_children] = n;
}

// ----------------------------------------

// LOC.py start
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

__device__ int8_t dagProperty(Env *env, DAG *dag) {
    LinkedList ll_visited;
    Set visited;
    setInit(&visited, &ll_visited);

    LinkedList path;
    llInit(&path);

    LinkedList work;
    llInit(&work);

    llAdd(&work, dag->root);
    while (work.size != 0) {
        Node *const next = llRemoveFirst(&work);
        if (!setContains(&visited, next)) {
            if (!nodeProperty(next, &path, &visited)) {
                return FALSE;
            }
        }
    }
    return (setSize(&visited) == env->num_of_nodes);
}

// __device__ void dagPrint(Node *root, char graph[]) {
//     LinkedList work;
//     llInit(&work);
//     llAdd(&work, root);
//     int index = 0;

//     while (work.size != 0) {
//         Node *current = llRemoveLast(&work);
//         if (current == NULL) {
//             graph[index++] = 'N';
//             graph[index++] = '-';
//             graph[index++] = '[';
//             graph[index++] = ']';
//             graph[index++] = '-';
//             continue;
//         } else {
//             graph[index++] = current->id + '0';
//             graph[index++] = '-';
//             if (current->num_of_children == 0) {
//                 graph[index++] = '[';
//                 graph[index++] = ']';
//                 graph[index++] = '-';
//             }
//         }

//         // add all children
//         for (int32_t i = 0; i < current->num_of_children; i++) {
//             llAdd(&work, current->children[i]);
//         }
//     }
//     graph[index++] = 'e';
//     graph[index] = '\0';
//     unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

//     printf("Graph of id %d: %s\n", idx, graph);
// }

__device__ void dagGenerate(Env *const env, DAG *const dag) {
    NodePool *const op = env->op;
    dag->root = op->getNew(op);

    for (int32_t i = 0; i < op->size; i++) {
        Node *const n = op->getObject(op, i);
        if (n != NULL) {
            n->id = i;
            const int8_t num_of_children = _choice(0, env->num_of_nodes - 1);
            n->num_of_children = num_of_children;
            for (int8_t j = 0; j < num_of_children; j++) {
                n->children[j] = op->getAny(op);
            }
        }
    }
}

__global__ void dagUdita(const int32_t bck_active, const int32_t size) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    NodePool op;
    initNodePool(&op, size, INCLUDE_NULL);

    Env env = {
        .num_of_nodes = size,
        .op = &op,
    };

    DAG dag = {
        .root = NULL,
    };

    dagGenerate(&env, &dag);
    const int8_t is_dag = dagProperty(&env, &dag);
    _countIf(is_dag);
    _ignoreIf(!is_dag);
    // if (is_dag) {
    //     char graph[50];
    //     dagPrint(dag.root, graph);
    // }

    #ifdef RUN_TEST
    // Node n  = {
    //     .num_of_children = 0,
    //     .id = size,
    //     .children = NULL,
    // };
    // dagAddNode(&env, &dag, idx, &n);
    dagProperty(&env, &dag);
    #endif
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
// LOC.py stop


int main(int argc, char* argv[]) {
    if (argc > 1) {
        return uditaMain(argc, argv, (void (*)(...)) dagUdita);
    } else {
        explore((void (*)(...)) testNodeProperty, NULL, 0);
        return 0;
    }
}
