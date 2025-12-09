# Contributing to Flower Recognition AI System

感谢您对 Flower Recognition AI System 项目的关注！我们欢迎所有形式的贡献。

## 🤝 如何贡献

### 报告Bug

如果您发现了Bug，请通过GitHub Issues报告：

1. 使用清晰的标题描述问题
2. 提供详细的复现步骤
3. 说明预期行为和实际行为
4. 提供环境信息（Python版本、PyTorch版本、GPU型号等）
5. 如果可能，附上错误日志和截图

**Bug报告模板**:
```markdown
**描述**
简要描述问题

**复现步骤**
1. 执行命令 '...'
2. 观察到 '...'
3. 发生错误 '...'

**预期行为**
应该发生什么

**实际行为**
实际发生了什么

**环境信息**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.8.10]
- PyTorch: [e.g., 2.0.1]
- CUDA: [e.g., 11.8]
- GPU: [e.g., NVIDIA RTX 3090]

**额外信息**
其他有用的信息
```

### 建议新功能

通过GitHub Issues提交功能请求：

1. 使用"Feature Request"标签
2. 清晰描述建议的功能
3. 说明为什么需要这个功能
4. 如果可能，提供实现思路

### 贡献代码

我们非常欢迎Pull Request！请遵循以下流程：

#### 1. Fork 项目

点击GitHub页面右上角的"Fork"按钮

#### 2. 克隆到本地

```bash
git clone https://github.com/YOUR_USERNAME/FlowerRecognition.git
cd FlowerRecognition
```

#### 3. 创建分支

```bash
# 创建并切换到新分支
git checkout -b feature/your-feature-name

# 或者修复bug
git checkout -b fix/your-bug-fix
```

分支命名规范：
- `feature/` - 新功能
- `fix/` - Bug修复
- `docs/` - 文档更新
- `refactor/` - 代码重构
- `test/` - 测试相关

#### 4. 设置开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install pytest black flake8 mypy
```

#### 5. 进行修改

- 遵循代码规范（见下文）
- 编写清晰的代码注释
- 添加必要的文档
- 如果修改了功能，更新相关文档

#### 6. 测试更改

```bash
# 运行现有测试（如果有）
pytest tests/

# 手动测试你的更改
python train.py  # 测试训练功能
python quickstart.py --checkpoint path/to/model.pt  # 测试推理功能
```

#### 7. 提交更改

```bash
# 添加修改的文件
git add .

# 提交（使用清晰的提交信息）
git commit -m "feat: add new data augmentation strategy"
```

提交信息规范：
- `feat:` - 新功能
- `fix:` - Bug修复
- `docs:` - 文档更新
- `style:` - 代码格式（不影响功能）
- `refactor:` - 重构
- `test:` - 测试
- `chore:` - 构建、工具等

#### 8. 推送到GitHub

```bash
git push origin feature/your-feature-name
```

#### 9. 创建Pull Request

1. 访问您fork的仓库
2. 点击"New Pull Request"
3. 填写PR描述，说明：
   - 做了什么修改
   - 为什么需要这些修改
   - 如何测试这些修改
   - 是否有相关Issue

**Pull Request模板**:
```markdown
## 描述
简要描述本次PR的目的和内容

## 类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 性能优化
- [ ] 代码重构

## 相关Issue
Closes #issue_number

## 修改内容
- 添加了...
- 修复了...
- 优化了...

## 测试
描述如何测试这些修改

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 添加了必要的注释
- [ ] 更新了相关文档
- [ ] 通过了所有测试
- [ ] 没有引入新的警告
```

## 📝 代码规范

### Python代码风格

遵循 [PEP 8](https://pep8.org/) 规范：

```python
# Good ✓
def train_model(config, data_loader, model, optimizer):
    """Train the model for one epoch.
    
    Args:
        config: Configuration object
        data_loader: Training data loader
        model: Neural network model
        optimizer: Optimizer instance
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        # Training code here
        pass
    
    return total_loss / len(data_loader)

# Bad ✗
def trainModel(cfg,dl,m,opt):
    m.train()
    l=0
    for i,(x,y) in enumerate(dl):
        pass
    return l/len(dl)
```

### 代码格式化

使用 Black 自动格式化：

```bash
# 格式化单个文件
black train.py

# 格式化整个项目
black .
```

### 类型提示

尽可能使用类型提示：

```python
from typing import Tuple, Optional
import torch
from torch import nn

def forward(
    self, 
    x: torch.Tensor, 
    return_features: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass with optional feature extraction."""
    pass
```

### 文档字符串

使用Google风格的docstring：

```python
def create_model(architecture: str, num_classes: int = 100) -> nn.Module:
    """Create a model instance.
    
    Args:
        architecture: Model architecture name (e.g., 'convnext_base')
        num_classes: Number of output classes. Defaults to 100.
        
    Returns:
        Initialized PyTorch model
        
    Raises:
        ValueError: If architecture is not supported
        
    Example:
        >>> model = create_model('convnext_base', num_classes=100)
        >>> output = model(torch.randn(1, 3, 600, 600))
    """
    pass
```

### 命名规范

```python
# 变量和函数：小写+下划线
train_loss = 0.0
learning_rate = 1e-4

def calculate_accuracy(predictions, targets):
    pass

# 类名：大驼峰
class FlowerDataset:
    pass

class ConvNextModel:
    pass

# 常量：全大写+下划线
MAX_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32

# 私有变量/方法：前缀下划线
_internal_state = {}

def _private_method():
    pass
```

## 🧪 测试

### 添加测试

如果添加新功能，请添加相应的测试：

```python
# tests/test_models.py
import pytest
import torch
from models import build_model

def test_model_output_shape():
    """Test model output has correct shape."""
    model = build_model('convnext_base', num_classes=100)
    x = torch.randn(2, 3, 600, 600)
    output = model(x)
    
    assert output.shape == (2, 100), f"Expected (2, 100), got {output.shape}"

def test_model_size_constraint():
    """Test model size meets competition requirement."""
    from models import get_model_size_mb
    
    model = build_model('convnext_base')
    size_mb = get_model_size_mb(model)
    
    assert size_mb <= 500, f"Model size {size_mb}MB exceeds 500MB limit"
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_models.py

# 运行特定测试函数
pytest tests/test_models.py::test_model_output_shape

# 显示详细输出
pytest -v

# 显示print输出
pytest -s
```

## 📚 文档

### 更新文档

如果修改了功能，请同步更新文档：

- **README.md** - 主要功能说明
- **docs/QUICKSTART.md** - 快速开始指南
- **docs/CONFIG_GUIDE.md** - 配置说明
- **docs/TRAINING_GUIDE.md** - 训练指南
- **docs/MODELS_GUIDE.md** - 模型文档

### 文档风格

- 使用清晰简洁的语言
- 提供实际的代码示例
- 包含必要的截图和图表
- 使用Markdown格式

## 🔍 代码审查

PR提交后，维护者会进行代码审查：

- 检查代码质量和规范
- 验证功能正确性
- 测试性能影响
- 审查文档完整性

请耐心等待审查，并根据反馈进行修改。

## 💡 最佳实践

1. **小步提交** - 每个PR专注于一个功能或修复
2. **清晰的提交信息** - 让人一眼看懂做了什么
3. **完善的测试** - 确保代码可靠性
4. **详细的文档** - 帮助他人理解使用
5. **遵循规范** - 保持代码一致性
6. **响应反馈** - 积极处理审查意见

## 📧 联系方式

有任何问题或建议，欢迎通过以下方式联系：

- **GitHub Issues**: [提交Issue](https://github.com/ironhxs/FlowerRecognition/issues)
- **Email**: 通过GitHub个人资料联系
- **Discussions**: 使用GitHub Discussions进行讨论

## 🙏 致谢

感谢所有贡献者的付出！您的贡献让这个项目变得更好。

---

再次感谢您的贡献！🌸
