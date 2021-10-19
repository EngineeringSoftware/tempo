#include <stdio.h>

#include "ts.h"
#include "../consts.h"

tree_set_entry* Env::nodeAlloc() {
    tree_set_entry *new_node = &(nodePool[nodePoolIndex]);
    nodePoolIndex++;
    if (nodePoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in nodePool, Index %d\n", nodePoolIndex);
    }

    return new_node;
}

tree_set::tree_set(Env *env)
    : root(nullptr), size(0), mod_count(0), RED(false), BLACK(true), env(env) {
}

bool tree_set::contains(int a_key) {
    return getEntry(a_key) != nullptr;
}

tree_set_entry* tree_set::getEntry(int key) {
    tree_set_entry* p = root;
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

bool tree_set::add(int a_key) {
    tree_set_entry* t = root;
    if (t == nullptr) {
        incrementSize();
        root = env->nodeAlloc();
        root->key = a_key;
        root->parent = nullptr;
        return false;
    }

    while (true) {
        if (a_key == t->key) {
            return true;
        } else {
            if (a_key < t->key) {
                if (t->left != nullptr) {
                    t = t->left;
                } else {
                    incrementSize();
                    t->left = env->nodeAlloc();
                    t->left->key = a_key;
                    t->left->parent = t;
                    fixAfterInsertion(t->left);
                    return false;
                }
            } else {
                if (t->right != nullptr) {
                    t = t->right;
                } else {
                    incrementSize();
                    t->right = env->nodeAlloc();
                    t->right->key = a_key;
                    t->right->parent = t;
                    fixAfterInsertion(t->right);
                    return false;
                }
            }
        }
    }
}

void tree_set::incrementSize() {
    mod_count++;
    size++;
}

bool tree_set::colorOf(tree_set_entry* p) {
    bool black = true;
    return (p == nullptr ? black : p->color);
}

tree_set_entry* tree_set::parentOf(tree_set_entry* p) {
    return (p == nullptr ? nullptr : p->parent);
}

void tree_set::setColor(tree_set_entry* p, bool c) {
    if (p != nullptr) {
        p->color = c;
    }
}

tree_set_entry* tree_set::leftOf(tree_set_entry* p) {
    return (p == nullptr) ? nullptr : p->left;
}

tree_set_entry* tree_set::rightOf(tree_set_entry* p) {
    return (p == nullptr) ? nullptr : p->right;
}

void tree_set::rotateLeft(tree_set_entry* p) {
    tree_set_entry* r = p->right;
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

void tree_set::rotateRight(tree_set_entry* p) {
    tree_set_entry* l = p->left;
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

void tree_set::fixAfterInsertion(tree_set_entry* entry) {
    tree_set_entry* x = entry;
    x->color = RED;
    while (x != nullptr && x != root && x->parent->color == RED) {
        if (parentOf(x) == leftOf(parentOf(parentOf(x)))) {
            tree_set_entry* y = rightOf(parentOf(parentOf(x)));
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
            tree_set_entry* y = leftOf(parentOf(parentOf(x)));
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

bool tree_set::remove(int a_key) {
    tree_set_entry* p = getEntry(a_key);
    if (p == nullptr) {
        return false;
    }

    deleteEntry(p);
    return true;
}

void tree_set::deleteEntry(tree_set_entry* p) {
    decrementSize();
    if (p->left != nullptr && p->right != nullptr) {
        tree_set_entry* s = successor(p);
        p->key = s->key;
        p = s;
    }

    tree_set_entry* replacement = (p->left != nullptr ? p->left : p->right);
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

        p->left = nullptr;
        p->right = nullptr;
        p->parent = nullptr;
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

void tree_set::fixAfterDeletion(tree_set_entry* entry) {
    tree_set_entry* x = entry;
    while (x != root && colorOf(x) == BLACK) {
        if (x == leftOf(parentOf(x))) {
            tree_set_entry* sib = rightOf(parentOf(x));
            if (colorOf(sib) == RED) {
                setColor(sib, BLACK);
                setColor(parentOf(x), RED);
                rotateLeft(parentOf(x));
                sib = rightOf(parentOf(x));
            }

            if (colorOf(leftOf(sib)) == BLACK && colorOf(rightOf(sib)) == BLACK) {
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
            tree_set_entry* sib = leftOf(parentOf(x));
            if (colorOf(sib) == RED) {
                setColor(sib, BLACK);
                setColor(parentOf(x), RED);
                rotateRight(parentOf(x));
                sib = leftOf(parentOf(x));
            }

            if (colorOf(rightOf(sib)) == BLACK && colorOf(leftOf(sib)) == BLACK) {
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

void tree_set::decrementSize() {
    mod_count++;
    size--;
}

tree_set_entry* tree_set::successor(tree_set_entry* t) {
    if (t == nullptr) {
        return nullptr;
    } else {
        if (t->right != nullptr) {
            tree_set_entry* p = t->right;
            while (p->left != nullptr) {
                p = p->left;
            }

            return p;
        } else {
            tree_set_entry* p = t->parent;
            tree_set_entry* ch = t;
            while (p != nullptr && ch == p->right) {
                ch = p;
                p = p->parent;
            }

            return p;
        }
    }
}
