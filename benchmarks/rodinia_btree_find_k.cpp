/**
 * Some portion of the source code is from the Rodinia benchmark suite.
 *
 * These are the original authors:
 * 2009; Amittai Aviram; entire code written in C;
 * 2010; Jordan Fix and Andrew Wilkes; code converted to CUDA;
 * 2011.10; Lukasz G. Szafaryn; code converted to portable form, to C, OpenMP, CUDA, PGI versions;
 * 2011.12; Lukasz G. Szafaryn; Split different versions for Rodinia.
 * 2011.12; Lukasz G. Szafaryn; code converted to OpenCL;
 * 2012.10; Ke Wang; Change it to non-interactive mode. Use command option read command from file. And also add output for easy verification among different platforms and devices.Merged into Rodinia main distribution 2.2.
 */

#include <boost/program_options.hpp>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sys/time.h>

#include "../device_hopper/core.h"

using namespace device_hopper;


// TODO the following define and structs should be in the common.h header
// but the source to source cl translation needs them in the benchmark

#define  DEFAULT_ORDER 256

typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER + 1];
	int  keys [DEFAULT_ORDER + 1];
	bool is_leaf;
	int num_keys;
} knode; 

typedef struct record {
	int value;
} record;

// ----------------------------------------------------------------------------------------------------------------------------


#include "./rodinia_src/btree/common.h"
#include "./rodinia_src/btree/kernel_gpu_cuda_wrapper.h"

#define PREFERRED_DEVICE GPU

using namespace std::chrono;

knode *knodes;
record *krecords;
char *mem;
long freeptr;
long malloc_size;
long size;
long maxheight;
int order = DEFAULT_ORDER;
node *queue = NULL;

void *kmalloc(int size) {
    //printf("size: %d, current offset: %p\n",size,freeptr);
    void *r = (void *) freeptr;
    freeptr += size;
    if (freeptr > malloc_size + (long) mem) {
        printf("Memory Overflow\n");
        exit(1);
    }
    return r;
}

void enqueue(node *new_node) {
    node *c;
    if (queue == NULL) {
        queue = new_node;
        queue->next = NULL;
    } else {
        c = queue;
        while (c->next != NULL) {
            c = c->next;
        }
        c->next = new_node;
        new_node->next = NULL;
    }
}

node *dequeue(void) {
    node *n = queue;
    queue = queue->next;
    n->next = NULL;
    return n;
}

node *insert(node *root, int key, int value) {
    record *pointer;
    node *leaf;

    /* The current implementation ignores duplicates. */
    if (find(root, key, false) != NULL)
        return root;

    /* Create a new record for the value. */
    pointer = make_record(value);

    /* Case: the tree does not exist yet. Start a new tree. */
    if (root == NULL)
        return start_new_tree(key, pointer);

    /* Case: the tree already exists. (Rest of function body.) */
    leaf = find_leaf(root, key, false);

    /* Case: leaf has room for key and pointer. */
    if (leaf->num_keys < order - 1) {
        leaf = insert_into_leaf(leaf, key, pointer);
        return root;
    }

    /* Case:  leaf must be split. */
    return insert_into_leaf_after_splitting(root, leaf, key, pointer);
}

long transform_to_cuda(node *root, bool verbose) {
    struct timeval one, two;
    double time;
    gettimeofday(&one, NULL);
    long max_nodes = (long) (pow(order, log(size) / log(order / 2.0) - 1) + 1);
    malloc_size = size * sizeof(record) + max_nodes * sizeof(knode);
    mem = (char *) malloc(malloc_size);
    if (mem == NULL) {
        printf("Initial malloc error\n");
        exit(1);
    }
    freeptr = (long) mem;

    krecords = (record *) kmalloc(size * sizeof(record));
    // printf("%d records\n", size);
    knodes = (knode *) kmalloc(max_nodes * sizeof(knode));
    // printf("%d knodes\n", max_nodes);

    queue = NULL;
    enqueue(root);
    node *n;
    knode *k;
    int i;
    long nodeindex = 0;
    long recordindex = 0;
    long queueindex = 0;
    knodes[0].location = nodeindex++;

    while (queue != NULL) {
        n = dequeue();
        k = &knodes[queueindex];
        k->location = queueindex++;
        k->is_leaf = n->is_leaf;
        k->num_keys = n->num_keys + 2;
        //start at 1 because 0 is set to INT_MIN
        k->keys[0] = INT_MIN;
        k->keys[k->num_keys - 1] = INT_MAX;
        for (i = k->num_keys; i < order; i++)k->keys[i] = INT_MAX;
        if (!k->is_leaf) {
            k->indices[0] = nodeindex++;
            // if(k->indices[0]>3953){
            // printf("ERROR: %d\n", k->indices[0]);
            // }
            for (i = 1; i < k->num_keys - 1; i++) {
                k->keys[i] = n->keys[i - 1];
                enqueue((node *) n->pointers[i - 1]);
                k->indices[i] = nodeindex++;
                // if(k->indices[i]>3953){
                // printf("ERROR 1: %d\n", k->indices[i]);
                // }
                //knodes[nodeindex].location = nodeindex++;
            }
            //for final point of n
            enqueue((node *) n->pointers[i - 1]);
        } else {
            k->indices[0] = 0;
            for (i = 1; i < k->num_keys - 1; i++) {
                k->keys[i] = n->keys[i - 1];
                krecords[recordindex].value = ((record *) n->pointers[i - 1])->value;
                k->indices[i] = recordindex++;
                // if(k->indices[i]>3953){
                // printf("ERROR 2: %d\n", k->indices[i]);
                // }
            }
        }

        k->indices[k->num_keys - 1] = queueindex;
        // if(k->indices[k->num_keys-1]>3953){
        // printf("ERROR 3: %d\n", k->indices[k->num_keys-1]);
        // }

        if (verbose) {
            printf("Successfully created knode with index %d\n", k->location);
            printf("Is Leaf: %d, Num Keys: %d\n", k->is_leaf, k->num_keys);
            printf("Pointers: ");
            for (i = 0; i < k->num_keys; i++)
                printf("%d | ", k->indices[i]);
            printf("\nKeys: ");
            for (i = 0; i < k->num_keys; i++)
                printf("%d | ", k->keys[i]);
            printf("\n\n");
        }
    }
    long mem_used = size * sizeof(record) + (nodeindex) * sizeof(knode);
    if (verbose) {
        for (i = 0; i < size; i++)
            printf("%d ", krecords[i].value);
        printf("\nNumber of records = %d, sizeof(record)=%d, total=%d\n", size, sizeof(record), size * sizeof(record));
        printf("Number of knodes = %d, sizeof(knode)=%d, total=%d\n", nodeindex, sizeof(knode),
               (nodeindex) * sizeof(knode));
        printf("\nDone Transformation. Mem used: %d\n", mem_used);
    }
    gettimeofday(&two, NULL);
    double oneD = one.tv_sec + (double) one.tv_usec * .000001;
    double twoD = two.tv_sec + (double) two.tv_usec * .000001;
    time = twoD - oneD;
    printf("Tree transformation took %f\n", time);

    return mem_used;
}


/* Utility function to give the height of the tree, which length in number of edges of the path from the root to any leaf. */
int height(node *root) {
    int h = 0;
    node *c = root;
    while (!c->is_leaf) {
        c = (node *) c->pointers[0];
        h++;
    }
    return h;
}

/* Traces the path from the root to a leaf, searching by key.  Displays information about the path if the verbose flag is set. Returns the leaf containing the given key. */
node *find_leaf(node *root, int key, bool verbose) {
    int i = 0;
    node *c = root;
    if (c == NULL) {
        if (verbose)
            printf("Empty tree.\n");
        return c;
    }
    while (!c->is_leaf) {
        if (verbose) {
            printf("[");
            for (i = 0; i < c->num_keys - 1; i++)
                printf("%d ", c->keys[i]);
            printf("%d] ", c->keys[i]);
        }
        i = 0;
        while (i < c->num_keys) {
            if (key >= c->keys[i])
                i++;
            else
                break;
        }
        if (verbose)
            printf("%d ->\n", i);
        c = (node *) c->pointers[i];
    }
    if (verbose) {
        printf("Leaf [");
        for (i = 0; i < c->num_keys - 1; i++)
            printf("%d ", c->keys[i]);
        printf("%d] ->\n", c->keys[i]);
    }
    return c;

}

/* Finds and returns the record to which a key refers. */
record *find(node *root, int key, bool verbose) {
    int i = 0;
    node *c = find_leaf(root, key, verbose);
    if (c == NULL)
        return NULL;
    for (i = 0; i < c->num_keys; i++)
        if (c->keys[i] == key)
            break;
    if (i == c->num_keys)
        return NULL;
    else
        return (record *) c->pointers[i];
}

/* Finds the appropriate place to split a node that is too big into two. */
int cut(int length) {
    if (length % 2 == 0)
        return length / 2;
    else
        return length / 2 + 1;
}

//======================================================================================================================================================150
// INSERTION
//======================================================================================================================================================150

/* Creates a new record to hold the value to which a key refers. */
record *make_record(int value) {
    record *new_record = (record *) malloc(sizeof(record));
    if (new_record == NULL) {
        perror("Record creation.");
        exit(EXIT_FAILURE);
    } else {
        new_record->value = value;
    }
    return new_record;
}

/* Creates a new general node, which can be adapted to serve as either a leaf or an internal node. */
node *make_node(void) {
    node *new_node;
    new_node = (node *) malloc(sizeof(node));
    if (new_node == NULL) {
        perror("Node creation.");
        exit(EXIT_FAILURE);
    }
    new_node->keys = (int *) malloc((order - 1) * sizeof(int));
    if (new_node->keys == NULL) {
        perror("New node keys array.");
        exit(EXIT_FAILURE);
    }
    new_node->pointers = (void **) malloc(order * sizeof(void *));
    if (new_node->pointers == NULL) {
        perror("New node pointers array.");
        exit(EXIT_FAILURE);
    }
    new_node->is_leaf = false;
    new_node->num_keys = 0;
    new_node->parent = NULL;
    new_node->next = NULL;
    return new_node;
}

/* Creates a new leaf by creating a node and then adapting it appropriately. */
node *make_leaf(void) {
    node *leaf = make_node();
    leaf->is_leaf = true;
    return leaf;
}

/* Helper function used in insert_into_parent to find the index of the parent's pointer to the node to the left of the key to be inserted. */
int get_left_index(node *parent, node *left) {
    int left_index = 0;
    while (left_index <= parent->num_keys &&
           parent->pointers[left_index] != left)
        left_index++;
    return left_index;
}

/* Inserts a new pointer to a record and its corresponding key into a leaf. Returns the altered leaf. */
node *insert_into_leaf(node *leaf, int key, record *pointer) {

    int i, insertion_point;

    insertion_point = 0;
    while (insertion_point < leaf->num_keys && leaf->keys[insertion_point] < key)
        insertion_point++;

    for (i = leaf->num_keys; i > insertion_point; i--) {
        leaf->keys[i] = leaf->keys[i - 1];
        leaf->pointers[i] = leaf->pointers[i - 1];
    }
    leaf->keys[insertion_point] = key;
    leaf->pointers[insertion_point] = pointer;
    leaf->num_keys++;
    return leaf;
}

/* Inserts a new key and pointer to a new record into a leaf so as to exceed the tree's order, causing the leaf to be split in half. */
node *insert_into_leaf_after_splitting(node *root,
                                       node *leaf,
                                       int key,
                                       record *pointer) {

    node *new_leaf;
    int *temp_keys;
    void **temp_pointers;
    int insertion_index, split, new_key, i, j;

    new_leaf = make_leaf();

    temp_keys = (int *) malloc(order * sizeof(int));
    if (temp_keys == NULL) {
        perror("Temporary keys array.");
        exit(EXIT_FAILURE);
    }

    temp_pointers = (void **) malloc(order * sizeof(void *));
    if (temp_pointers == NULL) {
        perror("Temporary pointers array.");
        exit(EXIT_FAILURE);
    }

    insertion_index = 0;
    while (leaf->keys[insertion_index] < key && insertion_index < order - 1)
        insertion_index++;

    for (i = 0, j = 0; i < leaf->num_keys; i++, j++) {
        if (j == insertion_index) j++;
        temp_keys[j] = leaf->keys[i];
        temp_pointers[j] = leaf->pointers[i];
    }

    temp_keys[insertion_index] = key;
    temp_pointers[insertion_index] = pointer;

    leaf->num_keys = 0;

    split = cut(order - 1);

    for (i = 0; i < split; i++) {
        leaf->pointers[i] = temp_pointers[i];
        leaf->keys[i] = temp_keys[i];
        leaf->num_keys++;
    }

    for (i = split, j = 0; i < order; i++, j++) {
        new_leaf->pointers[j] = temp_pointers[i];
        new_leaf->keys[j] = temp_keys[i];
        new_leaf->num_keys++;
    }

    free(temp_pointers);
    free(temp_keys);

    new_leaf->pointers[order - 1] = leaf->pointers[order - 1];
    leaf->pointers[order - 1] = new_leaf;

    for (i = leaf->num_keys; i < order - 1; i++)
        leaf->pointers[i] = NULL;
    for (i = new_leaf->num_keys; i < order - 1; i++)
        new_leaf->pointers[i] = NULL;

    new_leaf->parent = leaf->parent;
    new_key = new_leaf->keys[0];

    return insert_into_parent(root, leaf, new_key, new_leaf);
}

/* Inserts a new key and pointer to a node into a node into which these can fit without violating the B+ tree properties. */
node *insert_into_node(node *root,
                       node *n,
                       int left_index,
                       int key,
                       node *right) {

    int i;

    for (i = n->num_keys; i > left_index; i--) {
        n->pointers[i + 1] = n->pointers[i];
        n->keys[i] = n->keys[i - 1];
    }
    n->pointers[left_index + 1] = right;
    n->keys[left_index] = key;
    n->num_keys++;
    return root;
}

/* Inserts a new key and pointer to a node into a node, causing the node's size to exceed the order, and causing the node to split into two. */
node *insert_into_node_after_splitting(node *root,
                                       node *old_node,
                                       int left_index,
                                       int key,
                                       node *right) {
    int i, j, split, k_prime;
    node *new_node, *child;
    int *temp_keys;
    node **temp_pointers;

    /* First create a temporary set of keys and pointers
    * to hold everything in order, including
    * the new key and pointer, inserted in their
    * correct places.
    * Then create a new node and copy half of the
    * keys and pointers to the old node and
    * the other half to the new.
    */

    temp_pointers = (node **) malloc((order + 1) * sizeof(node *));
    if (temp_pointers == NULL) {
        perror("Temporary pointers array for splitting nodes.");
        exit(EXIT_FAILURE);
    }
    temp_keys = (int *) malloc(order * sizeof(int));
    if (temp_keys == NULL) {
        perror("Temporary keys array for splitting nodes.");
        exit(EXIT_FAILURE);
    }

    for (i = 0, j = 0; i < old_node->num_keys + 1; i++, j++) {
        if (j == left_index + 1) j++;
        temp_pointers[j] = (node *) old_node->pointers[i];
    }

    for (i = 0, j = 0; i < old_node->num_keys; i++, j++) {
        if (j == left_index) j++;
        temp_keys[j] = old_node->keys[i];
    }

    temp_pointers[left_index + 1] = right;
    temp_keys[left_index] = key;

    /* Create the new node and copy
    * half the keys and pointers to the
    * old and half to the new.
    */
    split = cut(order);
    new_node = make_node();
    old_node->num_keys = 0;
    for (i = 0; i < split - 1; i++) {
        old_node->pointers[i] = temp_pointers[i];
        old_node->keys[i] = temp_keys[i];
        old_node->num_keys++;
    }
    old_node->pointers[i] = temp_pointers[i];
    k_prime = temp_keys[split - 1];
    for (++i, j = 0; i < order; i++, j++) {
        new_node->pointers[j] = temp_pointers[i];
        new_node->keys[j] = temp_keys[i];
        new_node->num_keys++;
    }
    new_node->pointers[j] = temp_pointers[i];
    free(temp_pointers);
    free(temp_keys);
    new_node->parent = old_node->parent;
    for (i = 0; i <= new_node->num_keys; i++) {
        child = (node *) new_node->pointers[i];
        child->parent = new_node;
    }

    /* Insert a new key into the parent of the two
* nodes resulting from the split, with
* the old node to the left and the new to the right.
*/

    return insert_into_parent(root, old_node, k_prime, new_node);
}

/* Inserts a new node (leaf or internal node) into the B+ tree. Returns the root of the tree after insertion. */
node *
insert_into_parent(node *root,
                   node *left,
                   int key,
                   node *right) {

    int left_index;
    node *parent;

    parent = left->parent;

    /* Case: new root. */

    if (parent == NULL)
        return insert_into_new_root(left, key, right);

    /* Case: leaf or node. (Remainder of
* function body.)
*/

    /* Find the parent's pointer to the left
* node.
*/

    left_index = get_left_index(parent, left);


    /* Simple case: the new key fits into the node.
*/

    if (parent->num_keys < order - 1)
        return insert_into_node(root, parent, left_index, key, right);

    /* Harder case:  split a node in order
* to preserve the B+ tree properties.
*/

    return insert_into_node_after_splitting(root, parent, left_index, key, right);
}

/* Creates a new root for two subtrees and inserts the appropriate key into the new root. */
node *
insert_into_new_root(node *left,
                     int key,
                     node *right) {

    node *root = make_node();
    root->keys[0] = key;
    root->pointers[0] = left;
    root->pointers[1] = right;
    root->num_keys++;
    root->parent = NULL;
    left->parent = root;
    right->parent = root;
    return root;
}

/* First insertion: start a new tree. */
node *
start_new_tree(int key,
               record *pointer) {

    node *root = make_leaf();
    root->keys[0] = key;
    root->pointers[0] = pointer;
    root->pointers[order - 1] = NULL;
    root->parent = NULL;
    root->num_keys++;
    return root;
}

//======================================================================================================================================================150
// DELETION
//======================================================================================================================================================150

/* Utility function for deletion. Retrieves the index of a node's nearest neighbor (sibling) to the left if one exists.  If not (the node is the leftmost child), returns -1 to signify this special case. */
int
get_neighbor_index(node *n) {

    int i;

    /* Return the index of the key to the left
* of the pointer in the parent pointing
* to n.
* If n is the leftmost child, this means
* return -1.
*/
    for (i = 0; i <= n->parent->num_keys; i++)
        if (n->parent->pointers[i] == n)
            return i - 1;

    // Error state.
    printf("Search for nonexistent pointer to node in parent.\n");
    //printf("Node:  %#x\n", (unsigned int)n);
    exit(EXIT_FAILURE);
}

/*   */
node *
remove_entry_from_node(node *n,
                       int key,
                       node *pointer) {

    int i, num_pointers;

    // Remove the key and shift other keys accordingly.
    i = 0;
    while (n->keys[i] != key)
        i++;
    for (++i; i < n->num_keys; i++)
        n->keys[i - 1] = n->keys[i];

    // Remove the pointer and shift other pointers accordingly.
    // First determine number of pointers.
    num_pointers = n->is_leaf ? n->num_keys : n->num_keys + 1;
    i = 0;
    while (n->pointers[i] != pointer)
        i++;
    for (++i; i < num_pointers; i++)
        n->pointers[i - 1] = n->pointers[i];


    // One key fewer.
    n->num_keys--;

    // Set the other pointers to NULL for tidiness.
    // A leaf uses the last pointer to point to the next leaf.
    if (n->is_leaf)
        for (i = n->num_keys; i < order - 1; i++)
            n->pointers[i] = NULL;
    else
        for (i = n->num_keys + 1; i < order; i++)
            n->pointers[i] = NULL;

    return n;
}

/*   */
node *
adjust_root(node *root) {

    node *new_root;

    /* Case: nonempty root.
* Key and pointer have already been deleted,
* so nothing to be done.
*/

    if (root->num_keys > 0)
        return root;

    /* Case: empty root.
*/

    // If it has a child, promote
    // the first (only) child
    // as the new root.

    if (!root->is_leaf) {
        new_root = (node *) root->pointers[0];
        new_root->parent = NULL;
    }

        // If it is a leaf (has no children),
        // then the whole tree is empty.

    else
        new_root = NULL;

    free(root->keys);
    free(root->pointers);
    free(root);

    return new_root;
}

/* Coalesces a node that has become too small after deletion with a neighboring node that can accept the additional entries without exceeding the maximum. */
node *
coalesce_nodes(node *root,
               node *n,
               node *neighbor,
               int neighbor_index,
               int k_prime) {

    int i, j, neighbor_insertion_index, n_start, n_end, new_k_prime;
    node *tmp;
    bool split;

    /* Swap neighbor with node if node is on the
* extreme left and neighbor is to its right.
*/

    if (neighbor_index == -1) {
        tmp = n;
        n = neighbor;
        neighbor = tmp;
    }

    /* Starting point in the neighbor for copying
* keys and pointers from n.
* Recall that n and neighbor have swapped places
* in the special case of n being a leftmost child.
*/

    neighbor_insertion_index = neighbor->num_keys;

    /*
* Nonleaf nodes may sometimes need to remain split,
* if the insertion of k_prime would cause the resulting
* single coalesced node to exceed the limit order - 1.
* The variable split is always false for leaf nodes
* and only sometimes set to true for nonleaf nodes.
*/

    split = false;

    /* Case:  nonleaf node.
* Append k_prime and the following pointer.
* If there is room in the neighbor, append
* all pointers and keys from the neighbor.
* Otherwise, append only cut(order) - 2 keys and
* cut(order) - 1 pointers.
*/

    if (!n->is_leaf) {

        /* Append k_prime.
    */

        neighbor->keys[neighbor_insertion_index] = k_prime;
        neighbor->num_keys++;


        /* Case (default):  there is room for all of n's keys and pointers
    * in the neighbor after appending k_prime.
    */

        n_end = n->num_keys;

        /* Case (special): k cannot fit with all the other keys and pointers
    * into one coalesced node.
    */
        n_start = 0; // Only used in this special case.
        if (n->num_keys + neighbor->num_keys >= order) {
            split = true;
            n_end = cut(order) - 2;
        }

        for (i = neighbor_insertion_index + 1, j = 0; j < n_end; i++, j++) {
            neighbor->keys[i] = n->keys[j];
            neighbor->pointers[i] = n->pointers[j];
            neighbor->num_keys++;
            n->num_keys--;
            n_start++;
        }

        /* The number of pointers is always
    * one more than the number of keys.
    */

        neighbor->pointers[i] = n->pointers[j];

        /* If the nodes are still split, remove the first key from
    * n.
    */
        if (split) {
            new_k_prime = n->keys[n_start];
            for (i = 0, j = n_start + 1; i < n->num_keys; i++, j++) {
                n->keys[i] = n->keys[j];
                n->pointers[i] = n->pointers[j];
            }
            n->pointers[i] = n->pointers[j];
            n->num_keys--;
        }

        /* All children must now point up to the same parent.
    */

        for (i = 0; i < neighbor->num_keys + 1; i++) {
            tmp = (node *) neighbor->pointers[i];
            tmp->parent = neighbor;
        }
    }

        /* In a leaf, append the keys and pointers of
    * n to the neighbor.
    * Set the neighbor's last pointer to point to
    * what had been n's right neighbor.
    */

    else {
        for (i = neighbor_insertion_index, j = 0; j < n->num_keys; i++, j++) {
            neighbor->keys[i] = n->keys[j];
            neighbor->pointers[i] = n->pointers[j];
            neighbor->num_keys++;
        }
        neighbor->pointers[order - 1] = n->pointers[order - 1];
    }

    if (!split) {
        root = delete_entry(root, n->parent, k_prime, n);
        free(n->keys);
        free(n->pointers);
        free(n);
    } else
        for (i = 0; i < n->parent->num_keys; i++)
            if (n->parent->pointers[i + 1] == n) {
                n->parent->keys[i] = new_k_prime;
                break;
            }

    return root;

}

/* Redistributes entries between two nodes when one has become too small after deletion but its neighbor is too big to append the small node's entries without exceeding the maximum */
node *
redistribute_nodes(node *root,
                   node *n,
                   node *neighbor,
                   int neighbor_index,
                   int k_prime_index,
                   int k_prime) {

    int i;
    node *tmp;

    /* Case: n has a neighbor to the left.
* Pull the neighbor's last key-pointer pair over
* from the neighbor's right end to n's left end.
*/

    if (neighbor_index != -1) {
        if (!n->is_leaf)
            n->pointers[n->num_keys + 1] = n->pointers[n->num_keys];
        for (i = n->num_keys; i > 0; i--) {
            n->keys[i] = n->keys[i - 1];
            n->pointers[i] = n->pointers[i - 1];
        }
        if (!n->is_leaf) {
            n->pointers[0] = neighbor->pointers[neighbor->num_keys];
            tmp = (node *) n->pointers[0];
            tmp->parent = n;
            neighbor->pointers[neighbor->num_keys] = NULL;
            n->keys[0] = k_prime;
            n->parent->keys[k_prime_index] = neighbor->keys[neighbor->num_keys - 1];
        } else {
            n->pointers[0] = neighbor->pointers[neighbor->num_keys - 1];
            neighbor->pointers[neighbor->num_keys - 1] = NULL;
            n->keys[0] = neighbor->keys[neighbor->num_keys - 1];
            n->parent->keys[k_prime_index] = n->keys[0];
        }
    }

        /* Case: n is the leftmost child.
    * Take a key-pointer pair from the neighbor to the right.
    * Move the neighbor's leftmost key-pointer pair
    * to n's rightmost position.
    */

    else {
        if (n->is_leaf) {
            n->keys[n->num_keys] = neighbor->keys[0];
            n->pointers[n->num_keys] = neighbor->pointers[0];
            n->parent->keys[k_prime_index] = neighbor->keys[1];
        } else {
            n->keys[n->num_keys] = k_prime;
            n->pointers[n->num_keys + 1] = neighbor->pointers[0];
            tmp = (node *) n->pointers[n->num_keys + 1];
            tmp->parent = n;
            n->parent->keys[k_prime_index] = neighbor->keys[0];
        }
        for (i = 0; i < neighbor->num_keys; i++) {
            neighbor->keys[i] = neighbor->keys[i + 1];
            neighbor->pointers[i] = neighbor->pointers[i + 1];
        }
        if (!n->is_leaf)
            neighbor->pointers[i] = neighbor->pointers[i + 1];
    }

    /* n now has one more key and one more pointer;
* the neighbor has one fewer of each.
*/

    n->num_keys++;
    neighbor->num_keys--;

    return root;
}

/* Deletes an entry from the B+ tree. Removes the record and its key and pointer from the leaf, and then makes all appropriate changes to preserve the B+ tree properties. */
node *
delete_entry(node *root,
             node *n,
             int key,
             void *pointer) {

    int min_keys;
    node *neighbor;
    int neighbor_index;
    int k_prime_index, k_prime;
    int capacity;

    // Remove key and pointer from node.

    n = remove_entry_from_node(n, key, (node *) pointer);

    /* Case:  deletion from the root.
*/

    if (n == root)
        return adjust_root(root);


    /* Case:  deletion from a node below the root.
* (Rest of function body.)
*/

    /* Determine minimum allowable size of node,
* to be preserved after deletion.
*/

    min_keys = n->is_leaf ? cut(order - 1) : cut(order) - 1;

    /* Case:  node stays at or above minimum.
* (The simple case.)
*/

    if (n->num_keys >= min_keys)
        return root;

    /* Case:  node falls below minimum.
* Either coalescence or redistribution
* is needed.
*/

    /* Find the appropriate neighbor node with which
* to coalesce.
* Also find the key (k_prime) in the parent
* between the pointer to node n and the pointer
* to the neighbor.
*/

    neighbor_index = get_neighbor_index(n);
    k_prime_index = neighbor_index == -1 ? 0 : neighbor_index;
    k_prime = n->parent->keys[k_prime_index];
    neighbor = neighbor_index == -1 ? (node *) n->parent->pointers[1] :
               (node *) n->parent->pointers[neighbor_index];

    capacity = n->is_leaf ? order : order - 1;

    /* Coalescence. */

    if (neighbor->num_keys + n->num_keys < capacity)
        return coalesce_nodes(root, n, neighbor, neighbor_index, k_prime);

        /* Redistribution. */

    else
        return redistribute_nodes(root, n, neighbor, neighbor_index, k_prime_index, k_prime);
}

/*   */
void
destroy_tree_nodes(node *root) {
    int i;
    if (root->is_leaf)
        for (i = 0; i < root->num_keys; i++)
            free(root->pointers[i]);
    else
        for (i = 0; i < root->num_keys + 1; i++)
            destroy_tree_nodes((node *) root->pointers[i]);
    free(root->pointers);
    free(root->keys);
    free(root);
}

/**
* Verify the results calculated using the device hopper programming 
* model using the original benchmark cuda kernel.
*/
bool
verify_results(
        record *records,
        long records_mem,
        knode *knodes,
        long knodes_elem,
        long knodes_mem,

        int order,
        long maxheight,
        int count,

        long *currKnode,
        long *offset,
        int *keys,
        record *dhopper_ans) {

    // Compute reference results
    // Zero out buffers for intermediate results
    memset(currKnode, 0, count * sizeof(long));
    memset(offset, 0, count * sizeof(long));

    // OUTPUT: ans CPU allocation
    record *referenceAns = (record *) malloc(sizeof(record) * count);
    // OUTPUT: ans CPU initialization
    for (int i = 0; i < count; i++) referenceAns[i].value = -1;

    kernel_gpu_cuda_wrapper(
            records,
            records_mem,
            knodes,
            knodes_elem,
            knodes_mem,
            order,
            maxheight,
            count,
            currKnode,
            offset,
            keys,
            referenceAns);

    bool success = true;
    for (int i = 0; i < count; ++i) {
        if (referenceAns[i].value != dhopper_ans[i].value) {
            std::cout << "Error at index: " << i << " ";
            std::cout << "Expected value: "
                      << referenceAns[i].value
                      << " Computed value: "
                      << dhopper_ans[i].value << std::endl;
            success = false;
            break;
        }
    }

    free(referenceAns);

    return success;
}

#define REPOSITORY_PATH std::string(std::getenv("PLASTICITY_ROOT"))

DEVICE_HOPPER_MAIN(int argc, char* argv[]) {
    DEVICE_HOPPER_SETUP

    boost::program_options::options_description desc("Options");
    desc.add_options()("problem-size", boost::program_options::value<size_t>(), "Sample count");
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

    size_t problem_size = 0;
    if (vm.count("problem-size") == 0) {
        std::cerr << "Error: Problem size is missing." << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        problem_size = vm["problem-size"].as<size_t>();
    }

    std::string path_to_input_file;
    int count = 0;
    if (problem_size == 3) {
        count = 4500000;
        path_to_input_file = REPOSITORY_PATH + "/library/benchmarks/input_data/rodinia/b+tree/4500k.txt";
    } else if (problem_size == 2) {
        count = 3000000;
        path_to_input_file = REPOSITORY_PATH + "/library/benchmarks/input_data/rodinia/b+tree/3000k.txt";
    } else if (problem_size == 1) {
        count = 1500000;
        path_to_input_file = REPOSITORY_PATH + "/library/benchmarks/input_data/rodinia/b+tree/1500k.txt";
    } else {
        std::cerr << "Error: Unknown problem size" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Load inputs
    node *root = NULL;
    // open input file
    FILE *file_pointer = fopen(path_to_input_file.c_str(), "r");
    if (file_pointer == NULL) {
        std::cerr << "Error: Failure to open input file." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // get # of numbers in the file
    fscanf(file_pointer, "%d\n", &size);
    int input = 0;
    while (!feof(file_pointer)) {
        fscanf(file_pointer, "%d\n", &input);
        root = insert(root, input, input);
    }
    fclose(file_pointer);

    // NOTE DH: this is done to ease handling the tree data with
    // CUDA but it can also be seen as a flattening of the tree and is 
    // not necessarily CUDA specific.
    long mem_used = transform_to_cuda(root, 0);
    maxheight = height(root);
    printf("Max height: %d\n", maxheight);
    long rootLoc = (long) knodes - (long) mem;

    // Allocate memory
    // INPUT: records CPU allocation (setting pointer in mem variable)
    record *records_initial = (record *) mem;
    long records_elem = (long) rootLoc / sizeof(record);
    long records_mem = (long) rootLoc;
    printf("records_elem=%d, records_unit_mem=%d, records_mem=%d\n", (int) records_elem, (int) sizeof(record), (int) records_mem);

    // INPUT: knodes CPU allocation (setting pointer in mem variable)
    knode *knodes_initial = (knode *) ((long) mem + (long) rootLoc);
    long knodes_elem = ((long) (mem_used) - (long) rootLoc) / sizeof(knode);
    long knodes_mem = (long) (mem_used) - (long) rootLoc;
    printf("knodes_elem=%d, knodes_unit_mem=%d, knodes_mem=%d\n", (int) knodes_elem, (int) sizeof(knode), (int) knodes_mem);

    // NOTE DH: buffers are copied to independent memory locations to 
    // integrate better with the current device hopper API.
    // An alternative and likely prettier approach is not to use 
    // the common memory location in the first place.
    record *records = (record *) device_hopper::use_existing_buffer(records_elem, sizeof(record), records_initial);
    knode *knodes = (knode *) device_hopper::use_existing_buffer(knodes_elem, sizeof(knode), knodes_initial);

    // INPUT: currKnode CPU allocation
    long *currKnode = (long *) device_hopper::malloc(count, sizeof(long));
    // INPUT: offset CPU initialization
    memset(currKnode, 0, count * sizeof(long));

    // INPUT: offset CPU allocation
    long *offset = (long *) device_hopper::malloc(count, sizeof(long));
    // INPUT: offset CPU initialization
    memset(offset, 0, count * sizeof(long));

    // INPUT: keys CPU allocation
    int *keys = (int *) device_hopper::malloc(count, sizeof(int));
    // INPUT: keys CPU initialization
    for (int i = 0; i < count; i++) keys[i] = (rand() / (float) RAND_MAX) * size;

    // OUTPUT: ans CPU allocation
    record *ans = (record *) device_hopper::malloc(count, sizeof(record));
    // OUTPUT: ans CPU initialization 
    for (int i = 0; i < count; i++) ans[i].value = -1;

    // // Create parallel for
    parallel_for pf(0, count*order, [=]DEVICE_HOPPER_LAMBDA() {
        // private thread IDs
        int thid = GET_ITERATION_WITHIN_BATCH();
        int bid  = GET_BATCH_ID();

        // processtree levels
        int i;
        for(i = 0; i < maxheight; i++){

            // if value is between the two keys
            if((knodes[currKnode[bid]].keys[thid]) <= keys[bid] && (knodes[currKnode[bid]].keys[thid+1] > keys[bid])){
                // this conditional statement is inserted to avoid crush due to but in original code
                // "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
                // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
                if(knodes[offset[bid]].indices[thid] < knodes_elem){
                    offset[bid] = knodes[offset[bid]].indices[thid];
                }
            }

            device_hopper::batch_barrier();
        
            // set for next tree level
            if(thid==0){
                currKnode[bid] = offset[bid];
            }

            device_hopper::batch_barrier();
        }

        //At this point, we have a candidate leaf node which may contain
        //the target record.  Check each key to hopefully find the record
        if(knodes[currKnode[bid]].keys[thid] == keys[bid]){
            ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
        }
    });
    // Register buffers and specify access patterns
    pf.add_buffer_access_patterns(
        device_hopper::buf(knodes,    direction::IN, pattern::ALL_OR_ANY),
        device_hopper::buf(records,   direction::IN, pattern::ALL_OR_ANY),
        device_hopper::buf(currKnode, direction::IN_OUT, data_kind::INTERIM_RESULTS, pattern::SUCCESSIVE_SUBSECTIONS(1)),
        device_hopper::buf(offset,    direction::IN_OUT, data_kind::INTERIM_RESULTS, pattern::SUCCESSIVE_SUBSECTIONS(1)),
        device_hopper::buf(keys,      direction::IN,     pattern::SUCCESSIVE_SUBSECTIONS(1)),
        device_hopper::buf(ans,       direction::IN_OUT, pattern::SUCCESSIVE_SUBSECTIONS(1)));
    // Add scalar kernel parameters
    pf.add_scalar_parameters(maxheight, knodes_elem);
    // Set optional tuning parameters and call run()
    pf.opt_set_simple_indices(true).opt_set_batch_size(256).run();

    bool success = verify_results(
                        records,
                        records_mem,
                        knodes,
                        knodes_elem,
                        knodes_mem,
                        order,
                        maxheight,
                        count,
                        currKnode,
                        offset,
                        keys,
                        ans);
    
    free(currKnode);
    free(offset);
    free(keys);
    free(ans);
    free(mem);

    if(success) {
        std::cout << "Info: The results are correct" << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cout << "Error: The results are incorrect" << std::endl;
        return EXIT_FAILURE;
    }
}
