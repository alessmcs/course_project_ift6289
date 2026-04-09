from tree_sitter_language_pack import get_parser
parser = get_parser("java")
print(f"Parser: {parser}")
code = "public class A { public void main() {} }"
tree = parser.parse(code.encode("utf-8"))
print(f"Tree: {tree}")
print(f"Root: {tree.root_node}")
print(f"Type: {tree.root_node.type}")
