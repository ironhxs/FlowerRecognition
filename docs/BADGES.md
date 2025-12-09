# Badges and Shields Configuration

This document lists all the badges used in the README and how to customize them.

## Current Badges

### 1. Python Version
```markdown
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
```

### 2. PyTorch
```markdown
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
```

### 3. License
```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

### 4. Code Style
```markdown
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
```

## Additional Badges (Optional)

### Build Status
```markdown
[![Build Status](https://github.com/ironhxs/FlowerRecognition/workflows/CI/badge.svg)](https://github.com/ironhxs/FlowerRecognition/actions)
```

### Code Coverage
```markdown
[![codecov](https://codecov.io/gh/ironhxs/FlowerRecognition/branch/main/graph/badge.svg)](https://codecov.io/gh/ironhxs/FlowerRecognition)
```

### Documentation
```markdown
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/ironhxs/FlowerRecognition/tree/main/docs)
```

### GitHub Stars
```markdown
[![GitHub stars](https://img.shields.io/github/stars/ironhxs/FlowerRecognition?style=social)](https://github.com/ironhxs/FlowerRecognition/stargazers)
```

### GitHub Forks
```markdown
[![GitHub forks](https://img.shields.io/github/forks/ironhxs/FlowerRecognition?style=social)](https://github.com/ironhxs/FlowerRecognition/network/members)
```

### Issues
```markdown
[![GitHub issues](https://img.shields.io/github/issues/ironhxs/FlowerRecognition)](https://github.com/ironhxs/FlowerRecognition/issues)
```

### Pull Requests
```markdown
[![GitHub pull requests](https://img.shields.io/github/issues-pr/ironhxs/FlowerRecognition)](https://github.com/ironhxs/FlowerRecognition/pulls)
```

### Last Commit
```markdown
[![GitHub last commit](https://img.shields.io/github/last-commit/ironhxs/FlowerRecognition)](https://github.com/ironhxs/FlowerRecognition/commits/main)
```

### Contributors
```markdown
[![Contributors](https://img.shields.io/github/contributors/ironhxs/FlowerRecognition)](https://github.com/ironhxs/FlowerRecognition/graphs/contributors)
```

### Repository Size
```markdown
[![Repo Size](https://img.shields.io/github/repo-size/ironhxs/FlowerRecognition)](https://github.com/ironhxs/FlowerRecognition)
```

### Language Count
```markdown
[![Language Count](https://img.shields.io/github/languages/count/ironhxs/FlowerRecognition)](https://github.com/ironhxs/FlowerRecognition)
```

### Top Language
```markdown
[![Top Language](https://img.shields.io/github/languages/top/ironhxs/FlowerRecognition)](https://github.com/ironhxs/FlowerRecognition)
```

## Custom Badges

### Competition Badge
```markdown
[![Competition](https://img.shields.io/badge/Competition-2025%20National%20Challenge-success)](https://github.com/ironhxs/FlowerRecognition)
```

### Model Performance
```markdown
[![Accuracy](https://img.shields.io/badge/Accuracy-95.8%25-brightgreen)](https://github.com/ironhxs/FlowerRecognition)
[![Speed](https://img.shields.io/badge/Inference-<100ms-blue)](https://github.com/ironhxs/FlowerRecognition)
[![Size](https://img.shields.io/badge/Model%20Size-<500MB-orange)](https://github.com/ironhxs/FlowerRecognition)
```

### Technology Stack
```markdown
[![timm](https://img.shields.io/badge/timm-0.9.0-blue)](https://github.com/huggingface/pytorch-image-models)
[![Hydra](https://img.shields.io/badge/Hydra-1.3.0-blue)](https://hydra.cc/)
[![Albumentations](https://img.shields.io/badge/Albumentations-1.3.0-blue)](https://albumentations.ai/)
```

### Development Status
```markdown
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/ironhxs/FlowerRecognition)
[![Maintained](https://img.shields.io/badge/Maintained-Yes-green)](https://github.com/ironhxs/FlowerRecognition)
```

## Badge Styles

Shields.io supports different styles:

### Flat (Default)
```markdown
![Badge](https://img.shields.io/badge/label-message-color)
```

### Flat Square
```markdown
![Badge](https://img.shields.io/badge/label-message-color?style=flat-square)
```

### Plastic
```markdown
![Badge](https://img.shields.io/badge/label-message-color?style=plastic)
```

### For the Badge
```markdown
![Badge](https://img.shields.io/badge/label-message-color?style=for-the-badge)
```

### Social
```markdown
![Badge](https://img.shields.io/badge/label-message-color?style=social)
```

## Color Options

Common colors:
- `brightgreen`: Success, positive metrics
- `green`: Stable, maintained
- `yellowgreen`: Warning, intermediate
- `yellow`: Caution
- `orange`: Important
- `red`: Error, critical
- `blue`: Information, neutral
- `lightgrey`: Inactive, deprecated
- `success`: Equivalent to brightgreen
- `important`: Equivalent to orange
- `critical`: Equivalent to red
- `informational`: Equivalent to blue
- `inactive`: Equivalent to lightgrey

Hex colors:
```markdown
![Badge](https://img.shields.io/badge/label-message-3498db)
```

## Dynamic Badges

### GitHub API
```markdown
![Stars](https://img.shields.io/github/stars/ironhxs/FlowerRecognition)
![Forks](https://img.shields.io/github/forks/ironhxs/FlowerRecognition)
![Issues](https://img.shields.io/github/issues/ironhxs/FlowerRecognition)
```

### Custom Endpoint
```markdown
![Custom](https://img.shields.io/endpoint?url=YOUR_JSON_ENDPOINT)
```

JSON format:
```json
{
  "schemaVersion": 1,
  "label": "label",
  "message": "message",
  "color": "blue"
}
```

## Badge Generator Tools

1. **Shields.io**: https://shields.io/
   - Most popular badge service
   - Supports GitHub, npm, PyPI, etc.

2. **Badge Maker**: https://badge.fury.io/
   - Simple badge creation

3. **Custom Badge Maker**: https://badgen.net/
   - Alternative to shields.io

## Usage Tips

1. **Don't overuse badges**: 4-8 badges is usually enough
2. **Group related badges**: Put similar badges together
3. **Keep them updated**: Ensure version numbers are current
4. **Use meaningful badges**: Only show important information
5. **Consider mobile view**: Too many badges can look cluttered on mobile

## Recommended Badge Layout

```markdown
<div align="center">

# Project Title

[![Python](python-badge)](link)
[![Framework](framework-badge)](link)
[![License](license-badge)](link)
[![Build](build-badge)](link)

One-line description

[Link 1](#) | [Link 2](#) | [Link 3](#)

</div>
```

## Badge Maintenance

Update badges when:
- [ ] Python version requirement changes
- [ ] Major dependencies update
- [ ] License changes
- [ ] CI/CD configuration changes
- [ ] Project reaches milestones (stars, forks, etc.)

---

For more information, visit [Shields.io Documentation](https://shields.io/).
