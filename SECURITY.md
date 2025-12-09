# Security Policy

## Supported Versions

Currently supported versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of the Flower Recognition AI System seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do Not Public Disclose

Please **do not** open a public GitHub issue for security vulnerabilities.

### 2. Report Privately

Send an email to the maintainer through GitHub private messaging or email with:

- **Subject**: "Security Vulnerability Report - Flower Recognition"
- **Description**: Detailed description of the vulnerability
- **Steps to Reproduce**: How to reproduce the issue
- **Impact**: Potential impact of the vulnerability
- **Suggested Fix**: If you have ideas on how to fix it (optional)

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 5 business days
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next release cycle

### 4. Disclosure Process

1. Security issue is received and assigned to a handler
2. Problem is confirmed and severity is determined
3. Fix is prepared and tested
4. New release is published with security fix
5. Public disclosure is made in release notes

## Security Best Practices

When using this project:

### Data Security

- **Never commit sensitive data** (API keys, credentials) to the repository
- Use `.env` files for sensitive configuration (add to `.gitignore`)
- Ensure training data is from trusted sources

### Model Security

- **Verify model checksums** before using pretrained weights
- Be cautious when loading models from untrusted sources
- Use `torch.load(..., weights_only=True)` when possible

### Code Security

- Keep dependencies up to date
- Review code changes in Pull Requests carefully
- Use virtual environments to isolate dependencies
- Run code in sandboxed environments when testing

### Deployment Security

- Sanitize user inputs if deploying as a service
- Implement rate limiting for inference APIs
- Use HTTPS for all network communications
- Monitor for unusual inference patterns

## Known Security Considerations

### PyTorch Model Loading

- This project uses `torch.load()` which can execute arbitrary code
- Only load models from trusted sources
- Models in `results/checkpoints/` are created by this project and should be safe

### Data Input

- Image files are processed using PIL/Pillow
- Malformed images could potentially cause issues
- Input validation is performed in `FlowerDataset` class

## Dependencies

We regularly monitor and update dependencies for security vulnerabilities. You can check for outdated packages:

```bash
pip list --outdated
```

To update all dependencies:

```bash
pip install --upgrade -r requirements.txt
```

## Security Scanning

This project uses automated security scanning:

- **GitHub Dependabot**: Monitors for vulnerable dependencies
- **CodeQL**: Static code analysis (if enabled)

## Contact

For security concerns, please contact:

- **GitHub**: Open a security advisory on GitHub
- **Email**: Contact through GitHub profile

Thank you for helping keep Flower Recognition AI System secure! 🔒
