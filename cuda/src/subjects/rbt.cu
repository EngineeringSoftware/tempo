#include "rbt.h"
#include "../consts.h"

node* Env::nodeAlloc() {
    node *new_node = &(nodePool[nodePoolIndex]);
    nodePoolIndex++;
    if (nodePoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in nodePool, Index %d\n", nodePoolIndex);
    }

    return new_node;
}

Object* Env::objectAlloc() {
    Object *new_object = &(objectPool[objectPoolIndex]);
    objectPoolIndex++;
    if (objectPoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in objectPool, Index %d\n", objectPoolIndex);
    }

    return new_object;
}

int_red_black_tree::int_red_black_tree(Env *env)
    : size(0), root(nullptr), env(env) {
}

node* int_red_black_tree::parent(node* n) {
    return n == nullptr ? nullptr : n->parent;
}

node* int_red_black_tree::left(node* n) {
    return n == nullptr ? nullptr : n->left;
}

node* int_red_black_tree::right(node* n) {
    return n == nullptr ? nullptr : n->right;
}

bool int_red_black_tree::containsKey(int key) {
    return findNode(key) != nullptr;
}

Object* int_red_black_tree::get(int key) {
    node* x = findNode(key);
    return x == nullptr ? nullptr : x->value;
}

void int_red_black_tree::put(int key, Object* value) {
    node* x = env->nodeAlloc();
    x->key = key;
    x->value = value;
    treeInsert(x);
    size++;
}

Object* int_red_black_tree::remove(int key) {
    node* ret = treeDelete(findNode(key));
    if (ret == nullptr) {
        return nullptr;
    }

    size--;
    return ret->value;
}

bool int_red_black_tree::getColor(node* x) {
    return x == nullptr ? BLACK : x->color;
}

void int_red_black_tree::setColor(node* x, bool color) {
    if (x != nullptr) {
        x->color = color;
    }
}

node* int_red_black_tree::findNode(int key) {
    node* cur = root;
    while (cur != nullptr && key != cur->key) {
        if (key < cur->key) {
            cur = cur->left;
        } else {
            cur = cur->right;
        }
    }
    
    return cur;
}

void int_red_black_tree::leftRotate(node* x) {
    node* y = x->right;
    x->right = y->left;
    if (y->left != nullptr) {
        y->left->parent = x;
    }

    y->parent = x->parent;
    if (x->parent == nullptr) {
        root = y;
    } else {
        if (x == x->parent->left) {
            x->parent->left = y;
        } else {
            x->parent->right = y;
        }
    }

    y->left = x;
    x->parent = y;
}

void int_red_black_tree::rightRotate(node* x) {
    node* y = x->left;
    x->left = y->right;
    if (y->right != nullptr) {
        y->right->parent = x;
    }

    y->parent = x->parent;
    if (x->parent == nullptr) {
        root = y;
    } else {
        if (x == x->parent->right) {
            x->parent->right = y;
        } else {
            x->parent->left = y;
        }
    }

    y->right = x;
    x->parent = y;
}

void int_red_black_tree::treeInsert(node* z) {
    node* y = nullptr;
    node* x = root;
    while (x != nullptr) {
        y = x;
        if (z->key < x->key) {
            x = x->left;
        } else {
            if (z->key == x->key) {
                x->value = z->value;
                size--;
                return;
            } else {
                x = x->right;
            }
        }
    }
    z->parent = y;
    if (y == nullptr) {
        root = z;
    } else {
        if (z->key < y->key) {
            y->left = z;
        } else {
            y->right = z;
        }
    }

    z->left = nullptr;
    z->right = nullptr;
    z->color = RED;
    treeInsertFix(z);
}

void int_red_black_tree::treeInsertFix(node* z) {
    while (getColor(z->parent) == RED) {
        if (parent(z) == left(parent(parent(z)))) {
            node* y = right(parent(parent(z)));
            if (getColor(y) == RED) {
                setColor(parent(z), BLACK);
                setColor(y, BLACK);
                setColor(parent(parent(z)), RED);
                z = parent(parent(z));
            } else {
                if (z == right(parent(z))) {
                    z = parent(z);
                    leftRotate(z);
                }

                setColor(parent(z), BLACK);
                setColor(parent(parent(z)), RED);
                if (parent(parent(z)) != nullptr) {
                    rightRotate(parent(parent(z)));
                }
            }
        } else {
            node* y = left(parent(parent(z)));
            if (getColor(y) == RED) {
                setColor(parent(z), BLACK);
                setColor(y, BLACK);
                setColor(parent(parent(z)), RED);
                z = parent(parent(z));
            } else {
                if (z == left(parent(z))) {
                    z = parent(z);
                    rightRotate(z);
                }

                setColor(parent(z), BLACK);
                setColor(parent(parent(z)), RED);
                if (parent(parent(z)) != nullptr) {
                    leftRotate(parent(parent(z)));
                }
            }
        }
    }

    root->color = BLACK;
}

node* int_red_black_tree::treeDelete(node* z) {
    if (z == nullptr) {
        return nullptr;
    }

    node* x;
    node* y;
    if (z->left == nullptr || z->right == nullptr) {
        y = z;
    } else {
        y = getIOS(z);
    }

    if (y->left != nullptr) {
        x = y->left;
    } else {
        x = y->right;
    }

    if (x != nullptr) {
        x->parent = y->parent;
    }

    if (y->parent == nullptr) {
        root = x;
    } else {
        if (y == y->parent->left) {
            y->parent->left = x;
        } else {
            y->parent->right = x;
        }
    }

    if (y != z) {
        z->key = y->key;
        z->value = y->value;
    }

    if (getColor(y) == BLACK) {
        if (x == nullptr) {
            treeDeleteFix(y);
        } else {
            treeDeleteFix(x);
        }
    }

    return y;
}

void int_red_black_tree::treeDeleteFix(node* x) {
    while (x->parent != nullptr && getColor(x) == BLACK) {
        if (x == x->parent->left || x->parent->left == nullptr) {
            node* w = x->parent->right;
            if (w == nullptr) {
                return;
            }

            if (getColor(w) == RED) {
                w->color = BLACK;
                x->parent->color = RED;
                leftRotate(x->parent);
                w = x->parent->right;
            }

            if (getColor(w->left) == BLACK && getColor(w->right) == BLACK) {
                w->color = RED;
                x = x->parent;
            } else {
                if (getColor(w->right) == BLACK) {
                    w->left->color = BLACK;
                    rightRotate(w);
                    w = x->parent->right;
                }

                w->color = x->parent->color;
                x->parent->color = BLACK;
                if (w->right != nullptr) {
                    w->right->color = BLACK;
                }

                leftRotate(x->parent);
                x = root;
            }
        } else {
            node* w = x->parent->left;
            if (w == nullptr) {
                return;
            }

            if (getColor(w) == RED) {
                w->color = BLACK;
                x->parent->color = RED;
                rightRotate(x->parent);
                w = x->parent->left;
            }

            if (getColor(w->right) == BLACK && getColor(w->left) == BLACK) {
                w->color = RED;
                x = x->parent;
            } else {
                if (getColor(w->left) == BLACK) {
                    w->right->color = BLACK;
                    leftRotate(w);
                    w = x->parent->left;
                }

                w->color = x->parent->color;
                x->parent->color = BLACK;
                if (w->left != nullptr) {
                    w->left->color = BLACK;
                }

                rightRotate(x->parent);
                x = root;
            }
        }
    }

    x->color = BLACK;
}

node* int_red_black_tree::getIOS(node* z) {
    node* x = nullptr;
    node* y = nullptr;
    x = z->right;
    while (x != nullptr) {
        y = x;
        x = x->left;
    }

    return y;
}
