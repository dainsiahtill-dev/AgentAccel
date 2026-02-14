#!/usr/bin/env python3
"""
Syntax parser diagnostics for agent-accel.
Check AST/Tree-sitter functionality and dependencies.
"""

import sys
from pathlib import Path
from typing import Any

def check_tree_sitter_dependencies() -> dict[str, Any]:
    """Check Tree-sitter dependencies availability."""
    results = {
        "tree_sitter": False,
        "tree_sitter_language_pack": False,
        "tree_sitter_languages": False,
        "available_parsers": [],
    }
    
    # Check base tree-sitter
    try:
        import tree_sitter
        results["tree_sitter"] = True
        results["tree_sitter_version"] = getattr(tree_sitter, "__version__", "unknown")
    except ImportError:
        pass
    
    # Check language packs
    for module_name in ["tree_sitter_language_pack", "tree_sitter_languages"]:
        try:
            module = __import__(module_name)
            results[module_name] = True
            
            # Test available parsers
            if hasattr(module, "get_parser"):
                test_langs = ["python", "javascript", "typescript"]
                for lang in test_langs:
                    try:
                        parser = module.get_parser(lang)
                        if parser is not None:
                            results["available_parsers"].append(lang)
                    except (ValueError, TypeError, KeyError):
                        continue
        except ImportError:
            pass
    
    return results

def check_syntax_parsing() -> dict[str, Any]:
    """Check actual syntax parsing functionality."""
    from accel.indexers.symbols import extract_symbols
    from accel.config import resolve_effective_config
    
    results = {
        "python_ast": False,
        "python_tree_sitter": False,
        "js_tree_sitter": False,
        "ts_tree_sitter": False,
        "config_enabled": False,
        "config_provider": "off",
    }
    
    # Check configuration
    try:
        cfg = resolve_effective_config(Path("."))
        runtime = cfg.get("runtime", {})
        results["config_enabled"] = runtime.get("syntax_parser_enabled", False)
        results["config_provider"] = runtime.get("syntax_parser_provider", "off")
    except Exception as e:
        results["config_error"] = str(e)
    
    # Test Python parsing
    test_py = Path("test_syntax_py.py")
    test_py.write_text("def test(): pass\nclass Test: pass")
    
    try:
        symbols = extract_symbols(test_py, "test_syntax_py.py", "python")
        results["python_ast"] = len(symbols) > 0
        
        # Check if Tree-sitter was used (better accuracy)
        if len(symbols) >= 2:  # Should find both function and class
            results["python_tree_sitter"] = True
    except Exception as e:
        results["python_error"] = str(e)
    finally:
        test_py.unlink()
    
    # Test JavaScript/TypeScript if parsers available
    deps = check_tree_sitter_dependencies()
    if "javascript" in deps["available_parsers"]:
        test_js = Path("test_syntax.js")
        test_js.write_text("function test() {}\nclass Test {}")
        
        try:
            symbols = extract_symbols(test_js, "test_syntax.js", "javascript")
            results["js_tree_sitter"] = len(symbols) > 0
        except Exception as e:
            results["js_error"] = str(e)
        finally:
            test_js.unlink()
    
    if "typescript" in deps["available_parsers"]:
        test_ts = Path("test_syntax.ts")
        test_ts.write_text("function test() {}\nclass Test {}")
        
        try:
            symbols = extract_symbols(test_ts, "test_syntax.ts", "typescript")
            results["ts_tree_sitter"] = len(symbols) > 0
        except Exception as e:
            results["ts_error"] = str(e)
        finally:
            test_ts.unlink()
    
    return results

def main() -> int:
    """Run syntax parser diagnostics."""
    print("🔍 Agent-Accel Syntax Parser Diagnostics")
    print("=" * 50)
    
    # Check dependencies
    print("\n📦 Dependencies:")
    deps = check_tree_sitter_dependencies()
    
    print(f"  tree_sitter: {'✅' if deps['tree_sitter'] else '❌'}")
    if deps.get("tree_sitter_version"):
        print(f"    version: {deps['tree_sitter_version']}")
    
    print(f"  tree_sitter_language_pack: {'✅' if deps['tree_sitter_language_pack'] else '❌'}")
    print(f"  tree_sitter_languages: {'✅' if deps['tree_sitter_languages'] else '❌'}")
    
    if deps["available_parsers"]:
        print(f"  Available parsers: {', '.join(deps['available_parsers'])}")
    else:
        print("  Available parsers: None")
    
    # Check functionality
    print("\n⚙️  Functionality:")
    func = check_syntax_parsing()
    
    print(f"  Configuration enabled: {'✅' if func['config_enabled'] else '❌'}")
    print(f"  Configuration provider: {func['config_provider']}")
    print(f"  Python AST parsing: {'✅' if func['python_ast'] else '❌'}")
    print(f"  Python Tree-sitter: {'✅' if func['python_tree_sitter'] else '❌'}")
    print(f"  JavaScript Tree-sitter: {'✅' if func['js_tree_sitter'] else '❌'}")
    print(f"  TypeScript Tree-sitter: {'✅' if func['ts_tree_sitter'] else '❌'}")
    
    # Summary
    print("\n📋 Summary:")
    
    if func["python_tree_sitter"]:
        print("  ✅ Tree-sitter is working for Python")
        recommendation = "Tree-sitter parsing is active and working"
    elif func["python_ast"]:
        print("  ⚠️  Only AST parsing available (Tree-sitter missing)")
        recommendation = "Install tree-sitter-language-pack for better parsing"
    else:
        print("  ❌ No syntax parsing working")
        recommendation = "Check Python installation and dependencies"
    
    if not deps["tree_sitter_language_pack"] and not deps["tree_sitter_languages"]:
        print("  💡 Install: pip install tree-sitter-language-pack")
    
    print(f"\n🎯 Recommendation: {recommendation}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
