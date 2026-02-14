# Agent-Accel FAQ

## Quick Start

### 如何初始化新项目？

```bash
python -m accel.cli init --project .
```

这会创建 `accel.yaml` 配置文件。

### 如何检查项目健康状态？

```bash
python -m accel.cli doctor --project .
```

### 如何生成上下文包？

```bash
python -m accel.cli context --project . --task "implement feature X"
```

## 常见问题

### Q: 索引构建失败怎么办？

**A:** 检查以下几点：
1. 确保项目根目录有 `accel.yaml`
2. 运行 `python -m accel.cli doctor` 查看详细诊断
3. 检查是否有权限读取项目文件
4. 查看 `.harborpilot/logs/` 中的日志文件

### Q: 上下文包太大怎么办？

**A:** 调整预算限制：

```yaml
# accel.yaml
context:
  max_chars: 12000      # 默认 24000
  max_snippets: 30      # 默认 60
  top_n_files: 8        # 默认 12
```

### Q: 如何排除特定文件？

**A:** 在 `accel.yaml` 中添加排除模式：

```yaml
index:
  exclude:
    - "generated/**"
    - "*.min.js"
    - "vendor/**"
```

### Q: 验证命令超时怎么办？

**A:** 在 `accel.local.yaml` 中调整超时：

```yaml
runtime:
  verify_stall_timeout_seconds: 60
  verify_max_wall_time_seconds: 1800
```

### Q: 如何启用调试日志？

**A:** 设置环境变量：

```bash
export ACCEL_MCP_DEBUG=1
python -m accel.mcp_server
```

### Q: 支持哪些编程语言？

**A:** 内置支持：
- Python
- TypeScript/JavaScript

可以通过 `language_profile_registry` 扩展其他语言。

### Q: 如何清理缓存？

**A:** 删除运行时目录：

```bash
rm -rf .harborpilot/runtime/cache/
```

或使用 CLI：

```bash
python -m accel.cli index --project . --force
```

### Q: MCP 服务器启动失败？

**A:** 检查：
1. Python 版本 >= 3.11
2. 所有依赖已安装：`pip install -e .`
3. 端口未被占用
4. 查看 `.harborpilot/runtime/logs/` 中的 MCP 日志

### Q: 如何处理大仓库？

**A:** 建议配置：

```yaml
# accel.yaml
index:
  max_file_mb: 1           # 减小最大文件大小
  exclude:
    - "*.pb.go"            # 排除生成的代码
    - "**/node_modules/**"
    - "**/.git/**"

runtime:
  index_workers: 4         # 限制并行索引数
```

### Q: 如何与 HarborPilot 集成？

**A:** HarborPilot 会自动检测 agent-accel。确保：
1. 项目根目录有 `accel.yaml`
2. HarborPilot 配置中启用了 agent-accel MCP
3. 运行了 `hp_start_run` 启动治理流程

## 性能优化

### 提升索引速度

```yaml
runtime:
  index_workers: 8                    # 增加工作线程
  index_delta_compact_every: 500      # 减少压缩频率
```

### 优化上下文生成

```yaml
runtime:
  semantic_cache_enabled: true        # 启用语义缓存
  lexical_ranker_enabled: true        # 启用词法排名
```

### 减少内存使用

```yaml
context:
  max_chars: 12000
  max_snippets: 30
  snippet_radius: 20
```

## 故障排除

### 问题："No module named 'accel'"

**解决：**
```bash
pip install -e .
```

### 问题：JSON 解析错误

**解决：** 检查配置文件格式。支持 JSON 和 YAML，但必须是有效的。

### 问题：SQLite 数据库锁定

**解决：**
1. 确保没有多个进程同时访问
2. 删除 `.harborpilot/runtime/*.db` 文件
3. 重启 MCP 服务器

### 问题：tree-sitter 解析失败

**解决：** 某些文件可能不被 tree-sitter 支持。可以：
1. 在 `index.exclude` 中排除这些文件
2. 禁用语法解析：`syntax_parser_enabled: false`

## 最佳实践

1. **定期运行 doctor**: 每周运行一次 `accel doctor` 检查项目健康
2. **使用 .gitignore**: 将 `.harborpilot/` 添加到 `.gitignore`
3. **提交 accel.yaml**: 将配置文件提交到版本控制
4. **不提交 accel.local.yaml**: 本地配置包含敏感信息，应添加到 `.gitignore`
5. **监控缓存大小**: 定期清理 `.harborpilot/runtime/cache/`

## 获取帮助

- 查看日志：`.harborpilot/logs/`
- 检查运行时状态：`.harborpilot/runtime/`
- 阅读完整文档：`README.md`
- 提交 Issue: 在 GitHub 仓库提交问题

## 许可证

MIT License - 详见 LICENSE 文件
