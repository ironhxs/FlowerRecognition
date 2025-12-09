# Git Commit Message Convention

本项目遵循约定式提交（Conventional Commits）规范，以保持清晰的项目历史。

## 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 基本格式示例

```
feat(models): add EfficientNetV2-L support

- Implement EfficientNetV2-L model architecture
- Add corresponding config file
- Update model factory function

Closes #123
```

## Type（类型）

必须是以下之一：

### 主要类型

- **feat**: 新功能（feature）
  ```
  feat(training): add mixed precision training support
  feat(dataset): implement custom data augmentation pipeline
  feat(models): add Swin Transformer V2 architecture
  ```

- **fix**: Bug修复
  ```
  fix(dataset): resolve image loading error for RGBA images
  fix(training): correct gradient accumulation logic
  fix(inference): fix batch size handling in prediction
  ```

- **docs**: 文档变更
  ```
  docs(readme): update installation instructions
  docs(api): add docstrings to model classes
  docs(guide): improve training guide with examples
  ```

- **style**: 代码格式调整（不影响代码功能）
  ```
  style: apply black formatter to all Python files
  style(models): fix indentation and spacing
  style: remove trailing whitespaces
  ```

- **refactor**: 代码重构（既不是新增功能也不是修复bug）
  ```
  refactor(dataset): simplify data loading logic
  refactor(models): extract common model components
  refactor(utils): reorganize utility functions
  ```

- **perf**: 性能优化
  ```
  perf(training): optimize data loading with prefetch
  perf(inference): reduce model inference latency
  perf(dataset): cache augmentation transforms
  ```

- **test**: 测试相关
  ```
  test(models): add unit tests for model architectures
  test(dataset): add integration tests for data pipeline
  test: increase test coverage to 80%
  ```

- **chore**: 构建过程或辅助工具的变动
  ```
  chore: update dependencies to latest versions
  chore(ci): configure GitHub Actions workflow
  chore: add pre-commit hooks configuration
  ```

### 其他类型

- **build**: 构建系统或外部依赖变更
  ```
  build: update PyTorch to 2.1.0
  build(deps): bump albumentations from 1.3.0 to 1.3.1
  ```

- **ci**: CI配置文件和脚本变更
  ```
  ci: add automated testing workflow
  ci(github-actions): update Python version matrix
  ```

- **revert**: 回滚之前的提交
  ```
  revert: revert "feat: add experimental feature"
  
  This reverts commit abc123def456.
  Reason: Feature caused unexpected errors in production.
  ```

## Scope（范围）

可选，指定提交影响的范围：

- **models**: 模型相关
- **dataset**: 数据集相关
- **training**: 训练流程
- **inference**: 推理相关
- **config**: 配置文件
- **cli**: 命令行工具
- **docs**: 文档
- **utils**: 工具函数
- **tests**: 测试

示例：
```
feat(models): add ConvNeXt Tiny architecture
fix(dataset): handle missing image files gracefully
docs(training): update training strategy guide
```

## Subject（主题）

- 使用祈使句，现在时（"add" 不是 "added" 或 "adds"）
- 不要大写首字母
- 结尾不加句号
- 限制在50个字符以内
- 清晰描述做了什么

✅ 好的例子：
```
feat(models): add dropout layer to prevent overfitting
fix(training): resolve NaN loss issue during training
docs(readme): update quick start guide
```

❌ 不好的例子：
```
Added new model  # 没有使用现在时
Fix bug.  # 不够具体
Updated files  # 太模糊
FEAT: NEW FEATURE  # 大写不正确
```

## Body（正文）

可选，提供更详细的说明：

- 解释**为什么**做这个改变，而不仅仅是**做了什么**
- 与之前的行为对比
- 说明副作用或注意事项
- 每行限制在72个字符以内

示例：
```
feat(training): implement early stopping mechanism

Early stopping monitors validation loss and stops training
when no improvement is observed for a specified number of epochs.

This helps prevent overfitting and saves training time.

Configuration options:
- patience: number of epochs to wait (default: 10)
- min_delta: minimum change to qualify as improvement (default: 0.001)
```

## Footer（页脚）

可选，用于关联Issue或说明破坏性变更：

### 关闭Issue
```
Closes #123
Fixes #456
Resolves #789
```

### 破坏性变更
```
BREAKING CHANGE: configuration format has changed

The model configuration now uses nested structure instead of flat keys.
Users need to update their config files accordingly.

Before:
  model_name: convnext_base
  model_pretrained: true

After:
  model:
    name: convnext_base
    pretrained: true
```

### 引用相关Issue
```
Related to #123
See also #456
```

## 完整示例

### 示例 1: 添加新功能
```
feat(models): add support for model ensemble

Implement ensemble prediction by averaging predictions from
multiple models. This improves accuracy by 2-3% on validation set.

Features:
- Load multiple checkpoints
- Average logits or probabilities
- Support for weighted averaging

Closes #145
```

### 示例 2: 修复Bug
```
fix(dataset): prevent memory leak in data loader

The data loader was keeping references to images in memory
after processing. Now properly release memory after each batch.

This reduces memory usage by approximately 30% during training.

Fixes #234
```

### 示例 3: 性能优化
```
perf(training): optimize data augmentation pipeline

Cache augmentation transforms and use multi-processing
for image loading. This improves training speed by 40%.

Benchmarks:
- Before: 3.2s/epoch
- After: 1.9s/epoch

Related to #178
```

### 示例 4: 破坏性变更
```
feat(config)!: migrate to Hydra configuration system

BREAKING CHANGE: old YAML config format is no longer supported

The project now uses Hydra for configuration management.
All config files need to be migrated to the new format.

Migration guide available in docs/CONFIG_MIGRATION.md

Closes #201
```

## 提交频率建议

- **原子性提交**: 每个提交应该是一个逻辑上的完整单元
- **频繁提交**: 经常提交小的变更，而不是一次性提交大量代码
- **可回滚**: 每个提交都应该能够独立回滚

## Git提交最佳实践

### 1. 提交前检查

```bash
# 查看修改的文件
git status

# 查看具体改动
git diff

# 只添加相关文件
git add <specific-files>

# 而不是
git add .  # 可能包含不相关的文件
```

### 2. 使用交互式暂存

```bash
# 分别暂存不同的改动
git add -p
```

### 3. 写好提交信息

```bash
# 使用编辑器写详细的提交信息
git commit

# 简单提交
git commit -m "feat(models): add new model architecture"

# 带正文的提交
git commit -m "feat(models): add new model architecture" -m "Detailed description here"
```

### 4. 修改最近的提交

```bash
# 修改最近一次提交信息
git commit --amend

# 添加遗漏的文件到最近一次提交
git add forgotten-file.py
git commit --amend --no-edit
```

### 5. 保持提交历史整洁

```bash
# 在推送前整理提交
git rebase -i HEAD~3

# 合并多个小提交
# 在交互式rebase中使用 squash 或 fixup
```

## Git提交检查清单

在提交前确认：

- [ ] 提交信息遵循约定式提交格式
- [ ] Type和Scope正确
- [ ] Subject清晰且简洁
- [ ] 必要时添加了Body说明
- [ ] 关联了相关Issue
- [ ] 代码已测试
- [ ] 没有包含调试代码或临时文件
- [ ] 遵循项目代码规范
- [ ] 更新了相关文档

## 工具支持

### Commitizen

使用Commitizen工具辅助生成符合规范的提交信息：

```bash
# 安装
pip install commitizen

# 使用
cz commit
```

### Git Hooks

配置Git Hooks自动检查提交信息格式：

```bash
# .git/hooks/commit-msg
#!/bin/sh
commit_msg_file=$1
commit_msg=$(cat "$commit_msg_file")

# 检查提交信息格式
if ! echo "$commit_msg" | grep -qE "^(feat|fix|docs|style|refactor|perf|test|chore)(\(.+\))?: .+"; then
    echo "Error: Commit message does not follow conventional commits format"
    echo "Format: <type>(<scope>): <subject>"
    exit 1
fi
```

## 参考资源

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Angular Commit Guidelines](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)
- [Semantic Versioning](https://semver.org/)

---

遵循这些规范将帮助维护清晰的项目历史，使协作更加高效！
