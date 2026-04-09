from tree_sitter_language_pack import get_parser
parser = get_parser("java")
code = "public interface B { void test(); }"
tree = parser.parse(code.encode("utf-8"))
print(f"Root: {tree.root_node}")
for node in tree.root_node.children:
    print(f"Child type: {node.type}")
    if node.type == "interface_declaration":
        print("Found interface_declaration")
    for sub in node.children:
        print(f"  Sub child type: {sub.type}")
