#include "ll.h"
#include "../consts.h"

linked_list_node* Env::nodeAlloc() {
    linked_list_node *new_node = &(nodePool[nodePoolIndex]);
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

linked_list::linked_list(Env *env)
    : size(0), modCount(0), header(createHeaderNode()), env(env) {
}

bool linked_list::isEqualValue(const Object& value_1, const Object& value_2) {
    return value_1.id == value_2.id;
}

void linked_list::updateNode(linked_list_node* node, Object* value) {
    node->setValue(value);
}

linked_list_node* linked_list::createHeaderNode() {
    return env->nodeAlloc();
}

linked_list_node* linked_list::createNode(Object* value) {
    linked_list_node *node = env->nodeAlloc();
    node->value = value;
    return node;
}

void linked_list::addNodeBefore(linked_list_node* node, Object* value) {
    linked_list_node* new_node = createNode(value);
    addNode(new_node, node);
}

void linked_list::addNodeAfter(linked_list_node* node, Object* value) {
    linked_list_node* new_node = createNode(value);
    addNode(new_node, node->next);
}

void linked_list::addNode(linked_list_node* node_to_insert, linked_list_node* insert_before_node) {
    node_to_insert->next = insert_before_node;
    node_to_insert->previous = insert_before_node->previous;
    insert_before_node->previous->next = node_to_insert;
    size++;
    modCount++;
}

void linked_list::removeNode(linked_list_node* node) {
    node->previous->next = node->next;
    node->next->previous = node->previous;
    size--;
    modCount++;
    // delete node;
}

void linked_list::removeAllNodes() {
    // delete header;
    header = createHeaderNode();
    header->next = header;
    header->previous = header;
    size = 0;
    modCount++;
}

linked_list_node* linked_list::getNode(int index, bool end_marker_allowed) {
    if (index < 0) {
        return nullptr;
    }
    if (!end_marker_allowed && index == size) {
        return nullptr;
    }
    if (index > size) {
        return nullptr;
    }

    linked_list_node* node;
    if (index < (size / 2)) {
        node = header->next;
        for (int current_index = 0; current_index < size; current_index++) {
            node = node->next;
        }
    } else {
        node = header;
        for (int current_index = size; current_index > index; current_index--) {
            node = node->previous;
        }
    }

    return node;
} 

int linked_list::getSize() {
    return size;
}

bool linked_list::isEmpty() {
    return getSize() == 0;
}

Object* linked_list::get(int index) {
    linked_list_node* node = getNode(index, false);
    return node->getValue();
}

int linked_list::indexOf(const Object& value) {
    int i = 0;
    for (linked_list_node* node = header->next; node != header; node = node->next) {
        if (isEqualValue(*(node->getValue()), value)) {
            return i;
        }
        i++;
    }
    
    return -1;
}

int linked_list::lastIndexOf(const Object& value) {
    int i = size - 1;
    for (linked_list_node* node = header->previous; node != header; node = node->previous) {
        if (isEqualValue(*(node->getValue()), value)) {
            return i;
        }
        i--;
    }

    return -1;
}

bool linked_list::contains(const Object& value) {
    return indexOf(value) != -1;
}

bool linked_list::add(Object* value) {
    addLast(value);
    return true;
}

void linked_list::add(int index, Object* value) {
    linked_list_node* node = getNode(index, true);
    addNodeBefore(node, value);
}

Object* linked_list::removeIndex(int index) {
    linked_list_node* node = getNode(index, false);
    Object* old_value = env->objectAlloc();
    old_value->id = node->getValue()->id;
    removeNode(node);
    return old_value;
}

bool linked_list::remove(const Object& value) {
    for (linked_list_node* node = header->next; node != header; node = node->next) {
        if (isEqualValue(*(node->getValue()), value)) {
            removeNode(node);
            return true;
        }
    }

    return false;
}

Object* linked_list::set(int index, Object* value) {
    linked_list_node* node = getNode(index, false);
    Object* old_value = node->getValue();
    updateNode(node, value);
    return old_value;
}

void linked_list::clear() {
    removeAllNodes();
}

Object* linked_list::getFirst() {
    linked_list_node* node = header->next;
    if (node == header) {
        return nullptr;
    }

    return node->getValue();
}

Object* linked_list::getLast() {
    linked_list_node* node = header->previous;
    if (node == header) {
        return nullptr;
    }

    return node->getValue();
}

bool linked_list::addFirst(Object* value) {
    addNodeAfter(header, value);
    return true;
}

bool linked_list::addLast(Object* value) {
    addNodeBefore(header, value);
    return true;
}

Object* linked_list::removeFirst() {
    linked_list_node* node = header->next;
    if (node == header) {
        return nullptr;
    }
    Object* old_value = env->objectAlloc();
    old_value->id = node->getValue()->id;
    removeNode(node);

    return old_value;
}

Object* linked_list::removeLast() {
    linked_list_node* node = header->previous;
    if (node == header) {
        return nullptr;
    }
    Object* old_value = env->objectAlloc();
    old_value->id = node->getValue()->id;
    removeNode(node);

    return old_value;
}
