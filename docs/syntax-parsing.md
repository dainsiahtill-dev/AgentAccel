# AST/Tree-sitter 语法解析功能

agent-accel 支持 AST 和 Tree-sitter 两种语法解析方式，用于精确提取代码符号信息。

## 功能概述

### Python 语言
- **AST 解析**: 使用 Python 标准库，始终可用
- **Tree-sitter 解析**: 更精确的语法分析，需要安装额外依赖

### JavaScript/TypeScript 语言
- **正则表达式**: 基础模式匹配（默认回退）
- **Tree-sitter 解析**: 完整语法支持，推荐使用

## 安装依赖

### 基础安装
```bash
pip install -e .
```

### 启用语法解析（推荐）
```bash
pip install -e ".[syntax]"
```

或者手动安装：
```bash
pip install tree-sitter-language-pack
```

## 配置

在 `accel.local.yaml` 中启用语法解析：

```yaml
runtime:
  syntax_parser_enabled: true
  syntax_parser_provider: "tree_sitter"  # 或 "auto"
```

### 配置选项

- `syntax_parser_enabled`: 是否启用高级语法解析
- `syntax_parser_provider`: 解析器提供者
  - `"off"`: 禁用（仅使用基础方法）
  - `"auto"`: 自动选择最佳可用解析器
  - `"tree_sitter"`: 强制使用 Tree-sitter

## 使用示例

### 检查功能状态
```bash
python -m scripts.diagnostics.syntax_check
```

### 索引构建
启用语法解析后，索引构建会自动提取更精确的符号信息：

```bash
accel index build
```

### 上下文生成
更准确的符号信息会提升上下文生成的质量：

```bash
accel context "Add error handling to data processing" \
  --changed-files "src/processor.py"
```

## 支持的语言

| 语言 | AST | Tree-sitter | 正则回退 |
|------|-----|-------------|----------|
| Python | ✅ | ✅ | - |
| JavaScript | - | ✅ | ✅ |
| TypeScript | - | ✅ | ✅ |

## 性能对比

### Python
- **AST**: 快速，内置，功能完整
- **Tree-sitter**: 稍慢，但更精确的错误恢复

### JavaScript/TypeScript
- **正则**: 快速，但功能有限
- **Tree-sitter**: 较慢，但功能完整

## 故障排除

### Tree-sitter 不可用
```bash
# 检查依赖
python -m scripts.diagnostics.syntax_check

# 安装缺失依赖
pip install tree-sitter-language-pack
```

### 符号提取不准确
1. 确保启用了 Tree-sitter 解析
2. 检查文件编码是否为 UTF-8
3. 验证语法是否正确

### 性能问题
- 对于大型代码库，考虑禁用 Tree-sitter 以提升速度
- 使用 `max_file_mb` 配置限制处理的文件大小

## 开发者信息

### 扩展新语言
在 `accel/indexers/symbols.py` 中添加新语言支持：

1. 在 `_load_tree_sitter_parser()` 中添加语言别名
2. 创建新的符号提取函数
3. 在 `extract_symbols()` 中添加语言分支

### 调试
启用调试日志：
```bash
ACCEL_MCP_DEBUG=1 accel-mcp
```
