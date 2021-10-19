#include <stdio.h>

#include "tm.h"
#include "../consts.h"

entry* Env::nodeAlloc() {
    entry *new_node = &(nodePool[nodePoolIndex]);
    nodePoolIndex++;
    if (nodePoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in nodePool, Index %d\n", nodePoolIndex);
    }

    return new_node;
}

tree_map::tree_map(Env *env) : root(nullptr), size(0), env(env) {
}

void tree_map::incrementSize() {
    size++;
}

void tree_map::decrementSize() {
    size--;
}

int tree_map::getSize() {
    return size;
}

bool tree_map::containsKey(int key) {
    return getEntry(key) != nullptr;
}

entry* tree_map::getEntry(int key) {
    entry* p = root;
    while (p != nullptr) {
        if (key == p->key) {
            return p;
        } else {
            if (key < p->key) {
                p = p->left;
            } else {
                p = p->right;
            }
        }
    }

    return nullptr;
}

void tree_map::put(int key) {
    entry* t = root;
    if (t == nullptr) {
        incrementSize();
        root = env->nodeAlloc();
        root->key = key;
        root->parent = nullptr;
        return;
    }

    while (true) {
        if (key == t->key) {
            return;
        } else {
            if (key < t->key) {
                if (t->left != nullptr) {
                    t = t->left;
                } else {
                    incrementSize();
                    t->left = env->nodeAlloc();
                    t->left->key = key;
                    t->left->parent = t;
                    fixAfterInsertion(t->left);
                    return;
                }

            } else {
                if (t->right != nullptr) {
                    t = t->right;
                } else {
                    incrementSize();
                    t->right = env->nodeAlloc();
                    t->right->key = key;
                    t->right->parent = t;
                    fixAfterInsertion(t->right);
                    return;
                }
            }
        }
    }
}

void tree_map::remove(int key) {
    entry* p = getEntry(key);
    if (p == nullptr) {
        return;
    }

    deleteEntry(p);
    return;
}

entry* tree_map::successor(entry* t) {
    if (t == nullptr) {
        return nullptr;
    } else {
        if (t->right != nullptr) {
            entry* p = t->right;
            while (p->left != nullptr) {
                p = p->left;
            }
            return p;
        } else {
            entry* p = t->parent;
            entry* ch = t;
            while (p != nullptr && ch == p->right) {
                ch = p;
                p = p->parent;
            }
            return p;
        }
    }
}

bool tree_map::colorOf(entry* p) {
    return (p == nullptr ? BLACK : p->color);
}

entry* tree_map::parentOf(entry* p) {
    return (p == nullptr ? nullptr : p->parent);
}

void tree_map::setColor(entry* p, bool c) {
    if (p != nullptr)
        p->color = c;
}

entry* tree_map::leftOf(entry* p) {
    return (p == nullptr) ? nullptr : p->left;
}

entry* tree_map::rightOf(entry* p) {
    return (p == nullptr) ? nullptr : p->right;
}

void tree_map::rotateLeft(entry* p) {
    entry* r = p->right;
    p->right = r->left;
    if (r->left != nullptr) {
        r->left->parent = p;
    }

    r->parent = p->parent;
    if (p->parent == nullptr) {
        root = r;
    } else {
        if (p->parent->left == p) {
            p->parent->left = r;
        } else {
            p->parent->right = r;
        }
    }

    r->left = p;
    p->parent = r;
}

void tree_map::rotateRight(entry* p) {
    entry* l = p->left;
    p->left = l->right;
    if (l->right != nullptr) {
        l->right->parent = p;
    }

    l->parent = p->parent;
    if (p->parent == nullptr) {
        root = l;
    } else {
        if (p->parent->right == p) {
            p->parent->right = l;
        } else {
            p->parent->left = l;
        }
    }

    l->right = p;
    p->parent = l;
}

void tree_map::fixAfterInsertion(entry* x) {
    x->color = RED;
    while (x != nullptr && x != root && x->parent->color == RED) {
        if (parentOf(x) == leftOf(parentOf(parentOf(x)))) {
            entry* y = rightOf(parentOf(parentOf(x)));
            if (colorOf(y) == RED) {
                setColor(parentOf(x), BLACK);
                setColor(y, BLACK);
                setColor(parentOf(parentOf(x)), RED);
                x = parentOf(parentOf(x));
            } else {
                if (x == rightOf(parentOf(x))) {
                    x = parentOf(x);
                    rotateLeft(x);
                }

                setColor(parentOf(x), BLACK);
                setColor(parentOf(parentOf(x)), RED);
                if (parentOf(parentOf(x)) != nullptr) {
                    rotateRight(parentOf(parentOf(x)));
                }
            }
        } else {
            entry* y = leftOf(parentOf(parentOf(x)));
            if (colorOf(y) == RED) {
                setColor(parentOf(x), BLACK);
                setColor(y, BLACK);
                setColor(parentOf(parentOf(x)), RED);
                x = parentOf(parentOf(x));
            } else {
                if (x == leftOf(parentOf(x))) {
                    x = parentOf(x);
                    rotateRight(x);
                }

                setColor(parentOf(x), BLACK);
                setColor(parentOf(parentOf(x)), RED);
                if (parentOf(parentOf(x)) != nullptr) {
                    rotateLeft(parentOf(parentOf(x)));
                }
            }
        }
    }
    root->color = BLACK;
}

void tree_map::deleteEntry(entry* p) {
    decrementSize();
    if (p->left != nullptr && p->right != nullptr) {
        entry* s = successor(p);
        swapPosition(s, p);
    }

    entry* replacement = (p->left != nullptr ? p->left : p->right);
    if (replacement != nullptr) {
        replacement->parent = p->parent;
        if (p->parent == nullptr) {
            root = replacement;
        } else {
            if (p == p->parent->left) {
                p->parent->left = replacement;
            } else {
                p->parent->right = replacement;
            }
        }

        p->left = p->right = p->parent = nullptr;
        if (p->color == BLACK) {
            fixAfterDeletion(replacement);
        }
    } else {
        if (p->parent == nullptr) {
            root = nullptr;
        } else {
            if (p->color == BLACK) {
                fixAfterDeletion(p);
            }

            if (p->parent != nullptr) {
                if (p == p->parent->left) {
                    p->parent->left = nullptr;
                } else {
                    if (p == p->parent->right) {
                        p->parent->right = nullptr;
                    }
                }

                p->parent = nullptr;
            }
        }
    }

}

void tree_map::fixAfterDeletion(entry* x) {
    while (x != root && colorOf(x) == BLACK) {
        if (x == leftOf(parentOf(x))) {
            entry* sib = rightOf(parentOf(x));
            if (colorOf(sib) == RED) {
                setColor(sib, BLACK);
                setColor(parentOf(x), RED);
                rotateLeft(parentOf(x));
                sib = rightOf(parentOf(x));
            }

            if (colorOf(leftOf(sib)) == BLACK
                    && colorOf(rightOf(sib)) == BLACK) {
                setColor(sib, RED);
                x = parentOf(x);
            } else {
                if (colorOf(rightOf(sib)) == BLACK) {
                    setColor(leftOf(sib), BLACK);
                    setColor(sib, RED);
                    rotateRight(sib);
                    sib = rightOf(parentOf(x));
                }

                setColor(sib, colorOf(parentOf(x)));
                setColor(parentOf(x), BLACK);
                setColor(rightOf(sib), BLACK);
                rotateLeft(parentOf(x));
                x = root;
            }
        } else {
            entry* sib = leftOf(parentOf(x));
            if (colorOf(sib) == RED) {
                setColor(sib, BLACK);
                setColor(parentOf(x), RED);
                rotateRight(parentOf(x));
                sib = leftOf(parentOf(x));
            }

            if (colorOf(rightOf(sib)) == BLACK
                    && colorOf(leftOf(sib)) == BLACK) {
                setColor(sib, RED);
                x = parentOf(x);
            } else {
                if (colorOf(leftOf(sib)) == BLACK) {
                    setColor(rightOf(sib), BLACK);
                    setColor(sib, RED);
                    rotateLeft(sib);
                    sib = leftOf(parentOf(x));
                }

                setColor(sib, colorOf(parentOf(x)));
                setColor(parentOf(x), BLACK);
                setColor(leftOf(sib), BLACK);
                rotateRight(parentOf(x));
                x = root;
            }
        }
    }
    setColor(x, BLACK);
}

void tree_map::swapPosition(entry* x, entry* y) {
    entry* px = x->parent, *lx = x->left, *rx = x->right;
    entry* py = y->parent, *ly = y->left, *ry = y->right;
    bool xWasLeftChild = px != nullptr && x == px->left;
    bool yWasLeftChild = py != nullptr && y == py->left;
    if (x == py) {
        x->parent = y;
        if (yWasLeftChild) {
            y->left = x;
            y->right = rx;
        } else {
            y->right = x;
            y->left = lx;
        }
    } else {
        x->parent = py;
        if (py != nullptr) {
            if (yWasLeftChild) {
                py->left = x;
            } else {
                py->right = x;
            }
        }

        y->left = lx;
        y->right = rx;
    }

    if (y == px) {
        y->parent = x;
        if (xWasLeftChild) {
            x->left = y;
            x->right = ry;
        } else {
            x->right = y;
            x->left = ly;
        }
    } else {
        y->parent = px;
        if (px != nullptr) {
            if (xWasLeftChild) {
                px->left = y;
            } else {
                px->right = y;
            }
        }

        x->left = ly;
        x->right = ry;
    }

    if (x->left != nullptr) {
        x->left->parent = x;
    }

    if (x->right != nullptr) {
        x->right->parent = x;
    }

    if (y->left != nullptr) {
        y->left->parent = y;
    }

    if (y->right != nullptr) {
        y->right->parent = y;
    }

    bool c = x->color;
    x->color = y->color;
    y->color = c;
    if (root == x) {
        root = y;
    } else {
        if (root == y) {
            root = x;
        }
    }
}
