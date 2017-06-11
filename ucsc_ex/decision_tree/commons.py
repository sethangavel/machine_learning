from config import *


class DNode:
    def __init__(self, node_type, **kwargs):
        self.left = None
        self.right = None
        self.target = None
        if node_type == "RULE":
            self.type = "RULE"
            self.feature_index = kwargs['feature_idx']
            self.tau = kwargs['tau']
        else:
            self.type = "LEAF"
            self.target = kwargs['target']

    def is_leaf(self):
        return self.type == "LEAF"

    def visit(self, stats_dict, indent_level):
        if self.type == "LEAF":
            log_debug("{} [LEAF]: {}".format(" " * indent_level, self.target))
            stats_dict['{}'.format(self.target)] += 1
        else:
            log_debug("{} [RULE]: X{} / {}".format(" " * indent_level, self.feature_index, self.tau))


def traverse_tree(node, stats_dict, indent_level=0):
    if node is None:
        return
    node.visit(stats_dict, indent_level)
    traverse_tree(node.left, stats_dict, indent_level + 1)
    traverse_tree(node.right, stats_dict, indent_level + 1)


def log_debug(*args):
    if ENABLE_DEBUG_LOG:
        print(*args)


def log(*args):
    print(*args)

