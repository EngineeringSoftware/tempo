
#include "avlt.h"
#include "../consts.h"

Node* Env::nodeAlloc() {
    Node *new_node = &(nodeHeap[nodesIndex]);
    nodesIndex++;
    if (nodesIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in nodes, Index %d\n", nodesIndex);
    }

    return new_node;
}

Object* Env::objectAlloc() {
    Object *new_object = &(objectHeap[objectsIndex]);
    objectsIndex++;
    if (objectsIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in objects, Index %d\n", objectsIndex);
    }

    return new_object;
}

int_avl_tree_map::int_avl_tree_map(Env *env)
    : root(nullptr), size(0), env(env) {
}

bool int_avl_tree_map::containsKey(int key) {
    return findNode(key) != nullptr;
}

Object* int_avl_tree_map::get(int key) {
    Node* x = findNode(key);
    return x == nullptr ? nullptr : x->value;
}

void int_avl_tree_map::put(int key, Object* value) {
    Node* x = env->nodeAlloc();
    x->key = key;
    x->value = value;
    treeInsert(x);
}

Object* int_avl_tree_map::remove(int key) {
    Node* fnd = findNode(key);
    if (fnd == nullptr) {
        return nullptr;
    }

    Node* ret = treeDelete(fnd);
    size--;
    return ret->value;
}

int int_avl_tree_map::getSize() {
    return size;
}

int int_avl_tree_map::getHeight(Node *x) {
    return x == nullptr ? -1 : x->height;
}

int int_avl_tree_map::getBalance(Node *x) {
    return x == nullptr ? 0 : getHeight(x->left) - getHeight(x->right);
}

Node* int_avl_tree_map::findNode(int key) {
    Node* cur = root;
    while (cur != nullptr && key != cur->key) {
        if (key < cur->key) {
            cur = cur->left;
        } else {
            cur = cur->right;
        }
    }
    return cur;
}

Node* int_avl_tree_map::rightRotate(Node *x) {
    if (x == nullptr) {
        return nullptr;
    }

    Node* y = x->left;
    x->left = y->right;
    y->right = x;
    int x_left_height = getHeight(x->left);
    int x_right_height = getHeight(x->right);
    x->height = 1 + (x_left_height > x_right_height ? x_left_height : x_right_height);
    
    int y_left_height = getHeight(y->left);
    int x_height = getHeight(x);
    y->height = 1 + (y_left_height > x_height ? y_left_height : x_height);
    return y;
}

Node* int_avl_tree_map::leftRotate(Node *x) {
    if (x == nullptr) {
        return nullptr;
    }

    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    int x_left_height = getHeight(x->left);
    int x_right_height = getHeight(x->right);
    x->height = 1 + (x_left_height > x_right_height ? x_left_height : x_right_height);

    int y_right_height = getHeight(y->right);
    int x_height = getHeight(x);
    y->height = 1 + (y_right_height > x_height ? y_right_height : x_height);
    return y;
}

Node* int_avl_tree_map::rightLeftRotate(Node *x) {
    x->right = rightRotate(x->right);
    return leftRotate(x);
}

Node* int_avl_tree_map::leftRightRotate(Node *x) {
    x->left = leftRotate(x->left);
    return rightRotate(x);
}

void int_avl_tree_map::treeInsert(Node *x) {
    Node* fnd = findNode(x->key);
    if (fnd != nullptr) {
        fnd->key = x->key;
        return;
    }

    root = treeInsertRecur(x, root);
    size++;
}

Node* int_avl_tree_map::treeInsertRecur(Node *x, Node *cur) {
    if (cur == nullptr) {
        return x;
    }

    if (((int) x->key < cur->key)) {
        cur->left = treeInsertRecur(x, cur->left);
    } else {
        if (((int) x->key == cur->key)) {
            cur->value = x->value;
            return cur;
        } else {
            cur->right = treeInsertRecur(x, cur->right);
        }
    }

    cur->height = 1 + (getHeight(cur->left) > getHeight(cur->right) ? getHeight(cur->left) : getHeight(cur->right));
    int balance = getBalance(cur);
    if (balance == -2) {
        if (getBalance(cur->right) == -1) {
            return leftRotate(cur);
        } else {
            if (getBalance(cur->right) == 1) {
                return rightLeftRotate(cur);
            }
        }
    } else {
        if (balance == 2) {
            if (getBalance(cur->left) == 1) {
                return rightRotate(cur);
            } else {
                if (getBalance(cur->left) == -1) {
                    return leftRightRotate(cur);
                }
            }
        }
    }

    return cur;
}

Node* int_avl_tree_map::treeDelete(Node *x) {
    root = treeDeleteRecur(x, root);
    return x;
}

Node* int_avl_tree_map::treeDeleteRecur(Node *x, Node *cur) {
    if (((int) x->key < cur->key)) {
        cur->left = treeDeleteRecur(x, cur->left);
    } else {
        if (((int) x->key == cur->key)) {
            return afterDelete(cur);
        } else {
            cur->right = treeDeleteRecur(x, cur->right);
        }
    }

    return deleteFix(cur);
}

Node* int_avl_tree_map::deleteFix(Node *cur) {
    cur->height = 1 + (getHeight(cur->left) > getHeight(cur->right) ? getHeight(cur->left) : getHeight(cur->right));
    int balance = getBalance(cur);
    if (balance == -2) {
        if (getBalance(cur->right) <= 0) {
            return leftRotate(cur);
        } else {
            if (getBalance(cur->right) == 1) {
                return rightLeftRotate(cur);
            }
        }
    } else {
        if (balance == 2) {
            if (getBalance(cur->left) >= 0) {
                return rightRotate(cur);
            } else {
                if (getBalance(cur->left) == -1) {
                    return leftRightRotate(cur);
                }
            }
        }
    }

    return cur;
}

Node* int_avl_tree_map::afterDelete(Node *x) {
    if (x->left == nullptr && x->right == nullptr) {
        return nullptr;
    } else {
        if (x->left != nullptr && x->right != nullptr) {
            Node* z = getIOS(x);
            x->right = treeDeleteRecur(z, x->right);
            x->key = z->key;
            return deleteFix(x);
        } else {
            if (x->left != nullptr) {
                return x->left;
            } else {
                return x->right;
            }
        }
    }
}

Node* int_avl_tree_map::getIOS(Node *z) {
    Node *x, *y = nullptr;
    x = z->right;
    while (x != nullptr) {
        y = x;
        x = x->left;
    }
    return y;
}
