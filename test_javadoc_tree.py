from tree_sitter_language_pack import get_parser
parser = get_parser("java")
code = """
/**
 * Test Javadoc
 */
public class C {
    /**
     * Method Javadoc
     */
    public void m() {}
}
"""
tree = parser.parse(code.encode("utf-8"))
def print_tree(node, depth=0):
    print("  " * depth + f"{node.type} ({node.start_point}-{node.end_point})")
    for child in node.children:
        print_tree(child, depth + 1)

print_tree(tree.root_node)
